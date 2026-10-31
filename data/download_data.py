"""Download the Taiwanese Bankruptcy Prediction dataset from the UCI ML Repository.

Usage (from the repository root):
    python data/download_data.py

Writes data/company_bankruptcy.csv and checks that it matches the copy used in
the thesis: 6,819 companies, 96 columns (95 features + "Bankrupt?"), 220 bankrupt.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip"
OUT = Path(__file__).resolve().parent / "company_bankruptcy.csv"

EXPECTED_ROWS = 6819
EXPECTED_COLS = 96
EXPECTED_BANKRUPT = 220


def main() -> None:
    if OUT.exists():
        print(f"{OUT} already exists; skipping download.")
    else:
        print(f"Downloading {URL}")
        with urllib.request.urlopen(URL) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        csv_name = next(n for n in archive.namelist() if n.endswith(".csv"))
        OUT.write_bytes(archive.read(csv_name))
        print(f"Saved {csv_name} -> {OUT}")

    df = pd.read_csv(OUT)
    assert df.shape == (EXPECTED_ROWS, EXPECTED_COLS), f"unexpected shape {df.shape}"
    assert int(df["Bankrupt?"].sum()) == EXPECTED_BANKRUPT, "unexpected class counts"
    print(f"OK: {df.shape[0]} rows x {df.shape[1]} columns, "
          f"{EXPECTED_BANKRUPT} bankrupt ({EXPECTED_BANKRUPT / EXPECTED_ROWS:.2%}).")


if __name__ == "__main__":
    main()
