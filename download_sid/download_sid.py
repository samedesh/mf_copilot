import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://portal.amfiindia.com/spages/{id}.pdf"
START_ID = 1
END_ID = 15500          # latest IDs seen were ~14590; pad upward for newer filings
OUT_DIR = Path("amfi_sids")
MANIFEST = OUT_DIR / "manifest.csv"
WORKERS = 8             # keep modest; be polite to the server
TIMEOUT = 30

OUT_DIR.mkdir(exist_ok=True)


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=WORKERS))
    s.headers.update({"User-Agent": "Mozilla/5.0 (SID-downloader; personal research use)"})
    return s


def download(session: requests.Session, doc_id: int):
    """Returns (doc_id, status, size_bytes)."""
    path = OUT_DIR / f"{doc_id}.pdf"
    if path.exists() and path.stat().st_size > 0:
        return doc_id, "skipped_exists", path.stat().st_size

    url = BASE_URL.format(id=doc_id)
    try:
        r = session.get(url, timeout=TIMEOUT)
    except requests.RequestException as e:
        return doc_id, f"error:{type(e).__name__}", 0

    if r.status_code != 200:
        return doc_id, f"http_{r.status_code}", 0
    # Some servers return 200 with an HTML error page, so check the PDF magic bytes
    if not r.content.startswith(b"%PDF"):
        return doc_id, "not_pdf", 0

    path.write_bytes(r.content)
    return doc_id, "downloaded", len(r.content)


def main():
    session = make_session()
    ids = range(START_ID, END_ID + 1)
    rows = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(download, session, i) for i in ids]
        for n, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if n % 200 == 0:
                ok = sum(1 for r in rows if r[1] in ("downloaded", "skipped_exists"))
                print(f"{n}/{len(futures)} checked, {ok} PDFs, {time.time()-t0:.0f}s")

    rows.sort()
    with open(MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "status", "bytes"])
        w.writerows(rows)

    ok = sum(1 for r in rows if r[1] in ("downloaded", "skipped_exists"))
    print(f"Done. {ok} PDFs saved in {OUT_DIR}/ (see {MANIFEST.name})")


if __name__ == "__main__":
    main()