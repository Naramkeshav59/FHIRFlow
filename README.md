# FHIRFlow

**Make hospital lab data queryable across sources.** FHIRFlow loads FHIR R4
from two very different producers into one DuckDB warehouse, validates it,
and repairs the defect that breaks cross-source analytics: lab results coded
in a hospital's local vocabulary instead of LOINC.

| | |
| --- | --- |
| **Problem** | Hospital FHIR exports use local codes. In the MIMIC-IV FHIR demo, **0% of 107,727 lab results** are queryable by LOINC (498 local codes), so a query like "all creatinine results" across MIMIC and Synthea silently misses every MIMIC row. Mapping local codes to LOINC is normally weeks of expert work. |
| **What I built** | A FHIR → DuckDB pipeline with data-quality rules as queries, and an LLM code-repair layer: the LLM rewrites each local test into LOINC's vocabulary, NLM's LOINC search retrieves candidates, and the LLM picks one *from those candidates* or abstains. Uncertain picks go to human review. |
| **Impact** | On a 40-code development sample, the LLM layer maps **65%** of local lab codes to exactly the right LOINC code, vs **17.5%** for a rule-based baseline and 7.5% for plain search. Those codes cover **87%** of the sample's lab results, vs 66% with rules. |

## Results (measured)

Ground truth: MIT-LCP's curated MIMIC-IV lab → LOINC mapping
([`d_labitems_to_loinc.csv`](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/mapping)).
The mapper only sees what's in the FHIR data (label, specimen fluid, units,
reference range, example values), never the answer key.

**Warehouse** (`python -m fhirflow.ingest`, 15 s):

| Source | Lab rows | Distinct codes | Queryable by LOINC |
| --- | --- | --- | --- |
| MIMIC-IV FHIR demo | 107,727 | 498 | 0% |
| Synthea (111 patients) | 29,936 | 146 | 100% |

Validation rules flag 107,727 non-LOINC lab codes, 95,258 non-UCUM units and
12,469 lab results with no value in MIMIC.

**Mapping accuracy** (exact LOINC match; `python scripts/eval_mapping.py`):

40-code development sample (fixed seed; both LLM steps on `qwen/qwen3.8-27b`):

| Strategy | Exact match | Lab rows correctly coded | Right code retrieved |
| --- | --- | --- | --- |
| Naive: search the local label, take the top hit | 7.5% | 6.0% | 47.5% |
| Rules: unit → property, fluid → system, take the top hit | 17.5% | 66.2% | 42.5% |
| **LLM: rewrite queries, retrieve, pick or abstain** | **65.0%** | **86.6%** | **82.5%** |

All 448 labelled codes, baselines only: naive 6.5%, rules 17.0% exact match.

The LLM abstained on 7.5% of codes. When the right code was among its
candidates, it picked it 79% of the time. Auto-applying only where the LLM
and the rule strategy agree covers 20% of codes at 75% precision, so most
codes still go to a reviewer; a stronger gate is the next step.

### What the eval found

- **Retrieval is the ceiling, not the LLM.** Searching the local label
  directly puts the right code in the top 15 only 42% of the time, and
  ranks it first 6.5% of the time. Phrasing the query in LOINC's own
  vocabulary ("Creatinine Mass/volume Serum or Plasma") puts `2160-0` first.
  So the LLM's first job is query rewriting, not picking.
- **The LLM's self-reported confidence is uncalibrated.** It reports 0.95 on
  wrong answers, so confidence can't decide what to auto-apply. Agreement
  between the LLM and the independent rule-based strategy is the gate
  instead; everything else goes to review.
- **The LLM can't invent a code.** An answer outside the retrieved
  candidates is treated as an abstention (tested).

### Caveats

- The LLM numbers are on a 40-code development sample; the other 408
  labelled codes are held out and not yet run (free-tier API quota).
- Exact-match accuracy undercounts near-equivalents (e.g. "Monocytes
  [#/volume] in Blood" vs the reference's "…by Automated count").
- The LLM may have seen MIMIC or LOINC material in pretraining.

## Running it

```bash
python scripts/download_data.py          # Synthea bundles
# MIMIC-IV FHIR demo (open access): physionet.org/content/mimic-iv-fhir-demo
python -m fhirflow.ingest                # -> data/warehouse.duckdb
python scripts/eval_mapping.py --strategies naive rules            # free
python scripts/eval_mapping.py --strategies naive rules llm --limit 40
pytest
```

Needs `GROQ_API_KEY` in `.env` for the LLM strategy. NLM search results are
cached on disk, so re-runs are reproducible and free.

## Data

- MIMIC-IV Clinical Database Demo on FHIR (PhysioNet, open access).
  Johnson, A., et al. *MIMIC-IV on FHIR demo*. PhysioNet.
- Synthea synthetic patient records (no PHI).
- LOINC via the NLM Clinical Tables API. LOINC® is a registered trademark
  of Regenstrief Institute, Inc.
