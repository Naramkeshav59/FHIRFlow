"""Tests for the deterministic parts: FHIR parsing, validation rules, query
rules, and the guarantee that the LLM can't answer with an invented code."""
import duckdb
import pandas as pd

from fhirflow import ingest
from fhirflow.terminology import mapper
from fhirflow.terminology.loinc_search import Candidate


def _obs(**over):
    base = {
        "resourceType": "Observation", "id": "o1",
        "subject": {"reference": "Patient/p1"},
        "category": [{"coding": [{"code": "laboratory"}]}],
        "code": {"coding": [{"system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-d-labitems",
                             "code": "50912", "display": "Creatinine"}]},
        "valueQuantity": {"value": 1.1, "unit": "mg/dL",
                          "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-units"},
        "referenceRange": [{"low": {"value": 0.5}, "high": {"value": 1.2}}],
        "specimen": {"reference": "Specimen/s1"},
    }
    return {**base, **over}


def test_lab_row_flattens_fhir_and_resolves_specimen_fluid():
    row = ingest._lab_row("mimic", _obs(), {"s1": "Blood"})
    assert row["code"] == "50912" and row["display"] == "Creatinine"
    assert row["value"] == 1.1 and row["unit"] == "mg/dL"
    assert (row["ref_low"], row["ref_high"]) == (0.5, 1.2)
    assert row["patient_id"] == "p1" and row["fluid"] == "Blood"


def test_synthea_urn_references_resolve():
    row = ingest._lab_row("synthea", _obs(subject={"reference": "urn:uuid:abc"}), {})
    assert row["patient_id"] == "abc"


def test_validation_flags_each_defect():
    con = duckdb.connect()
    patients = pd.DataFrame([{"source": "mimic", "patient_id": "p1", "gender": "f", "birth_date": None}])  # noqa: F841
    loinc = _obs(id="o2", code={"coding": [{"system": "http://loinc.org", "code": "2160-0"}]},
                 valueQuantity={"value": 1.0, "unit": "mg/dL", "system": "http://unitsofmeasure.org"})
    orphan = _obs(id="o3", subject={"reference": "Patient/ghost"})
    labs = pd.DataFrame([ingest._lab_row("mimic", o, {}) for o in (_obs(), loinc, orphan)])  # noqa: F841
    con.execute("CREATE TABLE patients AS SELECT * FROM patients")
    con.execute("CREATE TABLE lab_observations AS SELECT * FROM labs")
    ingest.validate(con)
    issues = set(con.execute("SELECT obs_id, rule FROM validation_issues").fetchall())
    assert ("o1", "lab_not_loinc") in issues and ("o1", "unit_not_ucum") in issues
    assert not any(obs == "o2" for obs, _ in issues), "a clean LOINC/UCUM row has no issues"
    assert ("o3", "unknown_patient") in issues


def test_rule_query_uses_unit_and_fluid():
    lc = mapper.LocalCode("50912", "Creatinine", "Blood", "mg/dL", 0.5, 1.2)
    assert mapper.rule_query(lc) == "Creatinine Mass/volume Serum or Plasma"
    lc = mapper.LocalCode("50983", "Sodium", "Urine", "mEq/L", None, None)
    assert mapper.rule_query(lc) == "Sodium Moles/volume Urine"


def test_llm_cannot_answer_with_a_code_retrieval_did_not_return(monkeypatch):
    cands = [Candidate("2160-0", "Creatinine [Mass/volume] in Serum or Plasma")]
    monkeypatch.setattr(mapper, "search_many", lambda queries, n: cands)
    replies = iter([{"queries": ["creatinine serum"]},
                    {"loinc": "9999-9", "confidence": 0.99, "reason": "made up"}])
    monkeypatch.setattr(mapper, "_llm", lambda model, prompt: next(replies))
    m = mapper.map_llm(mapper.LocalCode("50912", "Creatinine", "Blood", "mg/dL", None, None))
    assert m.loinc is None and m.confidence == 0.0


def test_llm_pick_from_candidates_is_kept(monkeypatch):
    cands = [Candidate("2160-0", "Creatinine [Mass/volume] in Serum or Plasma")]
    monkeypatch.setattr(mapper, "search_many", lambda queries, n: cands)
    replies = iter([{"queries": ["creatinine serum"]},
                    {"loinc": "2160-0", "confidence": 0.9, "reason": "same test"}])
    monkeypatch.setattr(mapper, "_llm", lambda model, prompt: next(replies))
    m = mapper.map_llm(mapper.LocalCode("50912", "Creatinine", "Blood", "mg/dL", None, None))
    assert m.loinc == "2160-0" and m.confidence == 0.9
