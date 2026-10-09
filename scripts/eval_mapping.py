"""Score the three mapping strategies against MIT-LCP's curated
MIMIC-IV lab -> LOINC mapping (data/reference/d_labitems_to_loinc.csv).

    python scripts/eval_mapping.py --strategies naive rules      # free, all codes
    python scripts/eval_mapping.py --strategies llm --limit 40   # costs Groq quota

Predictions append to data/eval/<strategy>.jsonl one code at a time, so a
run stopped by the daily quota resumes where it left off.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fhirflow.ingest import DB_PATH  # noqa: E402
from fhirflow.terminology.mapper import AUTO_APPLY_CONFIDENCE, LocalCode, map_llm, map_naive, map_rules  # noqa: E402

REFERENCE = ROOT / "data" / "reference" / "d_labitems_to_loinc.csv"
OUT = ROOT / "data" / "eval"
STRATEGIES = {"naive": map_naive, "rules": map_rules, "llm": map_llm}


def local_codes(con) -> list[LocalCode]:
    """One record per MIMIC lab code, built only from the FHIR data."""
    df = con.execute("""
        SELECT code, any_value(display) AS label,
               mode(fluid) AS fluid, mode(unit) AS unit,
               mode(ref_low) AS ref_low, mode(ref_high) AS ref_high,
               list(value ORDER BY obs_id)[1:5] AS examples, count(*) AS n_rows
        FROM lab_observations WHERE source = 'mimic' GROUP BY code
        ORDER BY code  -- GROUP BY order isn't stable; the dev/test split depends on this
    """).df()
    return [LocalCode(r.code, r.label, r.fluid, r.unit,
                      None if pd.isna(r.ref_low) else r.ref_low,
                      None if pd.isna(r.ref_high) else r.ref_high,
                      [v for v in r.examples if v is not None and not pd.isna(v)], int(r.n_rows))
            for r in df.itertuples()]


def gold() -> dict[str, str]:
    ref = pd.read_csv(REFERENCE, dtype=str).rename(columns={"itemid (omop_source_code)": "itemid"})
    ref = ref[ref.omop_vocabulary_id.eq("LOINC") & ref.omop_concept_code.notna()]
    return dict(zip(ref.itemid, ref.omop_concept_code))


def run(strategy: str, codes: list[LocalCode]) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{strategy}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["local_code"]] = row
    with path.open("a", encoding="utf-8") as fh:
        for i, lc in enumerate(codes):
            if lc.code in done:
                continue
            row = STRATEGIES[strategy](lc).to_dict()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done[lc.code] = row
            if strategy == "llm":
                print(f"  {i + 1}/{len(codes)} {lc.code} {lc.label!r} -> {row['loinc']} ({row['confidence']})",
                      file=sys.stderr)
    return [done[lc.code] for lc in codes if lc.code in done]


def score(rows: list[dict], truth: dict[str, str], weight: dict[str, int]) -> dict:
    n = len(rows)
    correct = [r for r in rows if r["loinc"] == truth[r["local_code"]]]
    retrieved = [r for r in rows if truth[r["local_code"]] in r["candidates"]]
    w_total = sum(weight[r["local_code"]] for r in rows)
    out = {
        "codes": n,
        "accuracy": len(correct) / n,
        "row_weighted_accuracy": sum(weight[r["local_code"]] for r in correct) / w_total,
        "retrieval_recall": len(retrieved) / n,
    }
    if rows and rows[0]["strategy"] == "llm":
        auto = [r for r in rows if r["loinc"] and (r["confidence"] or 0) >= AUTO_APPLY_CONFIDENCE]
        out["abstained"] = sum(r["loinc"] is None for r in rows) / n
        out["auto_apply_share"] = len(auto) / n
        out["auto_apply_precision"] = (sum(r["loinc"] == truth[r["local_code"]] for r in auto) / len(auto)
                                       if auto else None)
        out["accuracy_when_retrieved"] = (sum(r["loinc"] == truth[r["local_code"]] for r in retrieved)
                                          / len(retrieved) if retrieved else None)
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=["naive", "rules"], choices=list(STRATEGIES))
    ap.add_argument("--limit", type=int, help="evaluate a fixed random sample of codes")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    truth = gold()
    codes = [lc for lc in local_codes(con) if lc.code in truth]
    random.Random(0).shuffle(codes)  # fixed order: --limit 40 is the same 40 codes every time
    if args.limit:
        codes = codes[:args.limit]
    weight = {lc.code: lc.n_rows for lc in codes}
    print(f"{len(codes)} labelled MIMIC lab codes ({sum(weight.values())} rows)", file=sys.stderr)

    rows = {s: run(s, codes) for s in args.strategies}
    results = {s: score(r, truth, weight) for s, r in rows.items()}
    print(pd.DataFrame(results).T.to_string())

    if {"llm", "rules"} <= set(rows):
        # The LLM's self-reported confidence turned out uncalibrated (0.95 on
        # wrong answers). Agreement between two independent strategies is a
        # gate that doesn't depend on the model grading itself.
        rule_top = {r["local_code"]: r["loinc"] for r in rows["rules"]}
        agree = [r for r in rows["llm"] if r["loinc"] and r["loinc"] == rule_top.get(r["local_code"])]
        n = len(rows["llm"])
        hits = sum(r["loinc"] == truth[r["local_code"]] for r in agree)
        print(f"\nauto-apply when LLM and rules agree: {len(agree)}/{n} codes "
              f"({len(agree) / n:.0%}), precision {hits / len(agree) if agree else 0:.0%}; "
              f"rest -> human review")


if __name__ == "__main__":
    main()
