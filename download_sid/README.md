# AMFI SID Downloader & Renamer (`download_sid`)

Utilities for bulk-downloading Indian Mutual Fund **Scheme Information Documents (SIDs)** from the AMFI portal and offline-renaming them from numeric document IDs into human-readable scheme filenames.

---

## Contents

| File | Purpose |
| :--- | :--- |
| `download_sid.py` | Multi-threaded downloader that fetches SID PDFs from AMFI (`1`–`15500`) into `amfi_sids/` and generates `manifest.csv`. |
| `download-listname.py` | 100% offline, idempotent renamer that extracts scheme names from the first 1–3 PDF pages and renames `{doc_id}.pdf` $\rightarrow$ `{clean_scheme_name}_{doc_id}.pdf`. |

---

## Prerequisites

- **Python 3.10+**
- Install required packages:
  ```powershell
  pip install requests pypdf
  ```
- **Internet access** to `https://portal.amfiindia.com` (for `download_sid.py` only; `download-listname.py` runs completely offline).

---

## 1. Download SID PDFs (`download_sid.py`)

Fetches PDFs from `https://portal.amfiindia.com/spages/{id}.pdf` for IDs `1` through `15500` using 8 worker threads (`WORKERS = 8`) and a 30-second timeout (`TIMEOUT = 30`).

### Key Features
- **Retry & Backoff:** Automatically retries transient HTTP errors (`429`, `500`, `502`, `503`, `504`) up to 4 times with exponential backoff (`backoff_factor=1.0`).
- **PDF Magic-Byte Validation:** Checks `b"%PDF"` so HTML error pages returned with HTTP 200 are rejected (`not_pdf`).
- **Resumable:** Skips already-downloaded non-empty `{doc_id}.pdf` files (`skipped_exists`).
- **Manifest Log:** Writes `amfi_sids/manifest.csv` with columns `doc_id,status,bytes`.

### Usage
Run from inside `download_sid/` (creates `download_sid/amfi_sids/`) or from your target working directory:
```powershell
python download_sid.py
```

### Configuration Constants (in `download_sid.py`)
- `START_ID = 1`, `END_ID = 15500` — Document ID range to scan.
- `OUT_DIR = Path("amfi_sids")` — Output folder for downloaded PDFs and `manifest.csv`.
- `WORKERS = 8` — Concurrent download threads.
- `TIMEOUT = 30` — Per-request timeout in seconds.

---

## 2. Rename Downloaded PDFs (`download-listname.py`)

Extracts the mutual fund scheme title from pages `0..2` of each PDF in `amfi_sids/` and renames files to:
```text
{clean_scheme_name}_{doc_id}.pdf
```

### Key Features
- **100% Offline & Idempotent:** Safe to run repeatedly without duplicating or compounding scheme names.
- **Boilerplate Filtering:** Strips SEBI disclaimers, stock exchange notices, offer dates, standalone AMC names, and leading/trailing page numbers.
- **Anchor-Based Extraction:** Prioritizes lines immediately before/after `SCHEME INFORMATION DOCUMENT`, with multi-line title stitching and multi-page fallback (cover/filing letter handling).
- **Windows-Safe Filenames:** Removes illegal characters (`\ / : * ? " < > |`), truncates cleanly at word boundaries ($\le$70 chars), and appends `_dup` if a filename collision occurs.

### Usage
```powershell
# Normal rename (reads amfi_sids/ in current dir, parent dir, or script dir)
python download-listname.py

# Preview renames without modifying any files
python download-listname.py --dry-run

# Revert all files back to numeric {doc_id}.pdf
python download-listname.py --revert

# Specify a custom PDF directory
python download-listname.py --dir path\to\amfi_sids
```
