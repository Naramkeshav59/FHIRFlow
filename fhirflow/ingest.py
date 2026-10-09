"""Load FHIR R4 from two very different producers into one DuckDB warehouse.

- MIMIC-IV FHIR demo: NDJSON, one resource per line, hospital-local codes
  (labs coded in `mimic-d-labitems`, units in `mimic-units`).
- Synthea: one transaction Bundle per patient, LOINC/SNOMED/UCUM throughout.

The point of loading both is that they *should* be queryable together and
aren't: a LOINC query finds Synthea's labs and silently misses every MIMIC
lab. validate() records that and the other defects as rows, so data quality
is a table you can query, not a log line.

    python -m fhirflow.ingest
"""
import gzip
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "warehouse.duckdb"

LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
LAB_CATEGORY = "laboratory"


def _ndjson(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def _ref_id(ref: dict | None) -> str | None:
    # "Patient/abc" -> "abc"; Synthea uses "urn:uuid:abc"
    if not ref or "reference" not in ref:
        return None
    return ref["reference"].split("/")[-1].removeprefix("urn:uuid:")


def _is_lab(obs: dict) -> bool:
    return any(c.get("code") == LAB_CATEGORY
               for cat in obs.get("category", []) for c in cat.get("coding", []))


def _lab_row(source: str, obs: dict, fluid_by_specimen: dict[str, str]) -> dict:
    coding = (obs.get("code", {}).get("coding") or [{}])[0]
    qty = obs.get("valueQuantity") or {}
    rr = (obs.get("referenceRange") or [{}])[0]
    return {
        "source": source,
        "obs_id": obs.get("id"),
        "patient_id": _ref_id(obs.get("subject")),
        "code_system": coding.get("system"),
        "code": coding.get("code"),
        "display": coding.get("display") or obs.get("code", {}).get("text"),
        "value": qty.get("value"),
        "unit": qty.get("unit") or qty.get("code"),
        "unit_system": qty.get("system"),
        "ref_low": (rr.get("low") or {}).get("value"),
        "ref_high": (rr.get("high") or {}).get("value"),
        "effective": obs.get("effectiveDateTime") or obs.get("issued"),
        "specimen_id": _ref_id(obs.get("specimen")),
        "fluid": fluid_by_specimen.get(_ref_id(obs.get("specimen")) or ""),
    }


def _patient_row(source: str, p: dict) -> dict:
    return {"source": source, "patient_id": p.get("id"), "gender": p.get("gender"),
            "birth_date": p.get("birthDate")}


def load_mimic() -> tuple[list[dict], list[dict]]:
    mimic = RAW / "mimic"
    fluid = {}
    for s in _ndjson(mimic / "MimicSpecimenLab.ndjson.gz"):
        coding = (s.get("type", {}).get("coding") or [{}])[0]
        fluid[s["id"]] = coding.get("code")
    patients = [_patient_row("mimic", p) for p in _ndjson(mimic / "MimicPatient.ndjson.gz")]
    labs = [_lab_row("mimic", o, fluid) for o in _ndjson(mimic / "MimicObservationLabevents.ndjson.gz")]
    return patients, labs


def load_synthea() -> tuple[list[dict], list[dict]]:
    patients, labs = [], []
    for path in sorted((RAW / "bundles").glob("*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        for entry in bundle.get("entry", []):
            r = entry.get("resource", {})
            if r.get("resourceType") == "Patient":
                patients.append(_patient_row("synthea", r))
            elif r.get("resourceType") == "Observation" and _is_lab(r):
                labs.append(_lab_row("synthea", r, {}))
    return patients, labs


def validate(con: duckdb.DuckDBPyConnection) -> None:
    """Each rule is a query; every violating row becomes an issue row."""
    rules = {
        # the defect the code-repair layer exists to fix
        "lab_not_loinc": "code_system IS DISTINCT FROM 'http://loinc.org'",
        "unit_not_ucum": f"value IS NOT NULL AND unit_system IS DISTINCT FROM '{UCUM}'",
        "missing_subject": "patient_id IS NULL",
        "unknown_patient": "patient_id NOT IN (SELECT patient_id FROM patients)",
        "no_value": "value IS NULL",
    }
    con.execute("CREATE OR REPLACE TABLE validation_issues "
                "(source VARCHAR, obs_id VARCHAR, rule VARCHAR)")
    for rule, where in rules.items():
        con.execute(f"INSERT INTO validation_issues "
                    f"SELECT source, obs_id, '{rule}' FROM lab_observations WHERE {where}")


def build(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    m_pat, m_lab = load_mimic()
    s_pat, s_lab = load_synthea()
    con = duckdb.connect(str(db_path))
    patients = pd.DataFrame(m_pat + s_pat)  # noqa: F841 -- read by DuckDB below
    labs = pd.DataFrame(m_lab + s_lab)  # noqa: F841
    con.execute("CREATE OR REPLACE TABLE patients AS SELECT * FROM patients")
    con.execute("CREATE OR REPLACE TABLE lab_observations AS SELECT * FROM labs")
    validate(con)
    return con


def summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        SELECT l.source,
               count(*) AS lab_rows,
               count(DISTINCT l.code) AS distinct_codes,
               round(100.0 * count(*) FILTER (WHERE l.code_system = 'http://loinc.org') / count(*), 1)
                   AS pct_loinc_queryable
        FROM lab_observations l GROUP BY l.source ORDER BY l.source
    """).df()


if __name__ == "__main__":
    con = build()
    print(summary(con).to_string(index=False), file=sys.stderr)
    print(con.execute("SELECT source, rule, count(*) n FROM validation_issues "
                      "GROUP BY ALL ORDER BY source, n DESC").df().to_string(index=False), file=sys.stderr)
