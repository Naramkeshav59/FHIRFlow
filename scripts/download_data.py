"""Download Synthea's pre-generated synthetic patient records as FHIR R4
bundles (one JSON bundle per patient), plus MIT-LCP's curated MIMIC-IV
lab -> LOINC mapping used as the eval's ground truth.

The MIMIC-IV FHIR demo itself is open access on PhysioNet; put its
*.ndjson.gz files in data/raw/mimic/.

    python scripts/download_data.py
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

REFERENCE = ("https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/"
             "mimic-iv/mapping/d_labitems_to_loinc.csv")
URL = "https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_fhir_latest.zip"
DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "synthea_fhir.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(DATA / "bundles")
    ref = DATA.parent / "reference" / "d_labitems_to_loinc.csv"
    ref.parent.mkdir(parents=True, exist_ok=True)
    if not ref.exists():
        urllib.request.urlretrieve(REFERENCE, ref)
    bundles = list((DATA / "bundles").rglob("*.json"))
    print(f"{len(bundles)} FHIR bundles in {DATA / 'bundles'}", file=sys.stderr)


if __name__ == "__main__":
    main()
