"""Download the Kaggle 'Cinema Tickets' dataset (needs a Kaggle API token in ~/.kaggle/kaggle.json
or KAGGLE_USERNAME / KAGGLE_KEY env vars). Manual alternative: download cinemaTicket_Ref.csv from
https://www.kaggle.com/datasets/arashnic/cinema-ticket and place it in data/raw/."""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

subprocess.run(["kaggle", "datasets", "download", "-d", "arashnic/cinema-ticket", "-p", str(RAW), "--unzip"],
               check=True)
f = RAW / "cinemaTicket_Ref.csv"
if not f.exists():
    sys.exit("cinemaTicket_Ref.csv not found after download")
print("SHA-256:", hashlib.sha256(f.read_bytes()).hexdigest())
