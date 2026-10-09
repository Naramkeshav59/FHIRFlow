"""Map a hospital-local lab code to LOINC.

Three strategies, so the eval can say whether the LLM earns its cost:

  naive  -- search the local label, take the top hit
  rules  -- rewrite the query into LOINC's vocabulary with fixed rules
            (unit -> property, fluid -> system), take the top hit
  llm    -- an LLM writes LOINC-style queries, retrieval returns candidates,
            a second LLM call picks one *from those candidates* or abstains

The LLM never invents a code: an answer that isn't in the retrieved set is
treated as an abstention. Abstentions and low-confidence picks are what a
human terminologist reviews; everything else can be applied automatically.

Only facts present in the FHIR data are used (label, specimen fluid, units,
reference range, example values) -- never the reference mapping, which is
the answer key.
"""
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv
from groq import BadRequestError, Groq, RateLimitError

from fhirflow.terminology.loinc_search import Candidate, search, search_many

load_dotenv()

# The reported results use qwen3.8-27b for both calls. They're separate
# settings because each Groq free-tier model has its own daily quota
# (1,000 requests, 200K tokens): a 400-code run can split the two calls
# across models.
REWRITE_MODEL = os.environ.get("REWRITE_MODEL", "qwen/qwen3.8-27b")
SELECT_MODEL = os.environ.get("SELECT_MODEL", "qwen/qwen3.8-27b")
AUTO_APPLY_CONFIDENCE = 0.8


@dataclass
class LocalCode:
    code: str
    label: str
    fluid: str | None
    unit: str | None
    ref_low: float | None
    ref_high: float | None
    examples: list = field(default_factory=list)
    n_rows: int = 0

    def describe(self) -> str:
        parts = [f'label "{self.label}"', f"specimen {self.fluid or 'unknown'}",
                 f"unit {self.unit or 'none'}"]
        if self.ref_low is not None or self.ref_high is not None:
            parts.append(f"reference range {self.ref_low}-{self.ref_high}")
        if self.examples:
            parts.append("example values " + ", ".join(map(str, self.examples[:5])))
        return "; ".join(parts)


@dataclass
class Mapping:
    local_code: str
    strategy: str
    loinc: str | None
    name: str | None = None
    confidence: float | None = None
    reason: str | None = None
    n_candidates: int = 0
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---- naive + rules ---------------------------------------------------------

def map_naive(lc: LocalCode) -> Mapping:
    hits = search(lc.label)
    top = hits[0] if hits else None
    return Mapping(lc.code, "naive", top and top.code, top and top.name,
                   n_candidates=len(hits), candidates=[c.code for c in hits])


UNIT_PROPERTY = [
    (r"^(mg|g|ug|ng|pg)/(dl|l|ml)$", "Mass/volume"),
    (r"^(meq|mmol|umol|nmol)/l$", "Moles/volume"),
    (r"^(iu|u|mu)/(l|ml)$", "Enzymatic activity/volume"),
    (r"^%$", "Fraction"),
    (r"^(k|m)/u?l$|^#/uL$|/uL$", "#/volume"),
    (r"^sec(onds)?$|^s$", "Time"),
    (r"^ratio$", "Ratio"),
]
FLUID_SYSTEM = {"Blood": "Serum or Plasma", "Urine": "Urine", "Cerebrospinal Fluid": "Cerebral spinal fluid",
                "Pleural": "Pleural fluid", "Ascites": "Peritoneal fluid", "Joint Fluid": "Synovial fluid",
                "Stool": "Stool", "Bone Marrow": "Bone marrow"}


def rule_query(lc: LocalCode) -> str:
    prop = next((p for pat, p in UNIT_PROPERTY if lc.unit and re.search(pat, lc.unit.strip(), re.I)), "")
    system = FLUID_SYSTEM.get(lc.fluid or "", lc.fluid or "")
    return " ".join(x for x in (lc.label, prop, system) if x)


def map_rules(lc: LocalCode) -> Mapping:
    hits = search(rule_query(lc))
    top = hits[0] if hits else None
    return Mapping(lc.code, "rules", top and top.code, top and top.name,
                   n_candidates=len(hits), candidates=[c.code for c in hits])


# ---- llm -------------------------------------------------------------------

REWRITE_PROMPT = """You map hospital lab codes to LOINC. Write search queries for a LOINC
search engine, phrased the way LOINC names tests: component, property, system
(e.g. "Creatinine [Mass/volume] in Serum or Plasma", "Hemoglobin A1c/Hemoglobin.total in Blood").
Use the unit to infer the property and the specimen to infer the system.
Expand abbreviations. Return JSON: {"queries": [3 distinct queries, most likely first]}.

Local lab test: {desc}"""

SELECT_PROMPT = """You map hospital lab codes to LOINC.

Local lab test: {desc}

Candidate LOINC codes:
{cands}

Pick the single candidate that is the same test: same analyte, same specimen
system, and a property consistent with the unit. Prefer the general test over
variants for a specific timing or method unless the label says otherwise.
Hospital chemistry, drug-level and serology tests on a "Blood" specimen are run
on serum or plasma, so prefer "in Serum or Plasma" for them; cell counts,
blood gases and hemoglobin measures stay "in Blood".
If no candidate is the same test, answer null.
Return JSON: {{"loinc": "<code from the list or null>", "confidence": <0-1>, "reason": "<one sentence>"}}"""

_client = None


def _llm(model: str, prompt: str) -> dict:
    global _client
    _client = _client or Groq(max_retries=6)
    for attempt in range(6):
        try:
            resp = _client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0, max_tokens=1500,
            )
            return json.loads(resp.choices[0].message.content)
        except RateLimitError:
            time.sleep(15 * (attempt + 1))
        except (json.JSONDecodeError, ValueError, BadRequestError):
            # BadRequestError here is Groq's json_validate_failed: the model
            # emitted empty or invalid JSON. A sampling fluke -- retry once,
            # then abstain rather than crash a 400-code run.
            if attempt >= 1:
                return {}
    return {}


def map_llm(lc: LocalCode, n_per_query: int = 12) -> Mapping:
    desc = lc.describe()
    queries = [q for q in _llm(REWRITE_MODEL, REWRITE_PROMPT.replace("{desc}", desc)).get("queries", [])
               if isinstance(q, str)][:3]
    # the rule-based query costs no tokens and often finds what the LLM's miss
    cands: list[Candidate] = search_many(queries + [rule_query(lc), lc.label], n_per_query)
    if not cands:
        return Mapping(lc.code, "llm", None, reason="no candidates retrieved")
    listing = "\n".join(f"{c.code}: {c.name}" for c in cands)
    pick = _llm(SELECT_MODEL, SELECT_PROMPT.format(desc=desc, cands=listing))
    by_code = {c.code: c for c in cands}
    code = pick.get("loinc")
    if code not in by_code:
        # an invented or garbled code is an abstention, never an answer
        code = None
    return Mapping(
        lc.code, "llm", code, by_code[code].name if code else None,
        confidence=float(pick.get("confidence") or 0) if code else 0.0,
        reason=pick.get("reason"), n_candidates=len(cands), candidates=list(by_code),
    )
