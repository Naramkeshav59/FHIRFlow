"""LOINC candidate retrieval via NLM's Clinical Tables API (free, no key).

Every response is cached on disk keyed by the query, so an eval run is
reproducible and re-runs cost nothing. The cache is also what keeps the LLM
honest: the mapper may only answer with a code that retrieval returned.
"""
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

URL = "https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search"
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "loinc_search"


@dataclass(frozen=True)
class Candidate:
    code: str
    name: str


def search(query: str, n: int = 15) -> list[Candidate]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (hashlib.sha1(f"{n}|{query}".encode()).hexdigest() + ".json")
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        for attempt in range(4):
            try:
                resp = requests.get(URL, params={"terms": query, "maxList": n,
                                                 "df": "LOINC_NUM,LONG_COMMON_NAME"}, timeout=20)
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        rows = resp.json()[3] or []
        path.write_text(json.dumps(rows), encoding="utf-8")
    return [Candidate(code, name) for code, name in rows]


def search_many(queries: list[str], n: int = 15) -> list[Candidate]:
    """Union of several queries' results, first-seen order, deduplicated."""
    seen: dict[str, Candidate] = {}
    for q in queries:
        for c in search(q, n):
            seen.setdefault(c.code, c)
    return list(seen.values())
