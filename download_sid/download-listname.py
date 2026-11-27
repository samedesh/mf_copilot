"""Offline SID File Renamer.
Safely extracts mutual fund scheme names from SIDs in 'amfi_sids/' and renames files to:
  {clean_scheme_name}_{doc_id}.pdf

Features:
- 100% Offline (no API or internet required)
- Idempotent: safe to run multiple times without duplicating or compounding names
- Strips legal/SEBI boilerplate, stock exchange disclaimers, offer dates, and leading page numbers
- Skips scheme category/regulatory descriptions in favor of the actual scheme title
- Prioritizes lines directly before/after 'SCHEME INFORMATION DOCUMENT'
- Multi-page fallback (checks pages 0..2 if page 0 is a cover/filing letter)
- Options:
    python download-listname.py             # Normal rename
    python download-listname.py --dry-run   # Preview renames without modifying files
    python download-listname.py --revert    # Revert all files back to pure {doc_id}.pdf
"""
from __future__ import annotations
import argparse
import logging
import re
import sys
import time
from pathlib import Path

# Suppress noisy font parsing warnings from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)
from pypdf import PdfReader

# Resolve amfi_sids directory whether run from root or inside download_sid/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUT_DIR = Path("amfi_sids")
if not OUT_DIR.exists():
    OUT_DIR = PROJECT_ROOT / "amfi_sids"
if not OUT_DIR.exists():
    OUT_DIR = SCRIPT_DIR / "amfi_sids"

# Standalone AMC brand names (not specific scheme titles)
AMC_ONLY = re.compile(
    r"^(?:reliance|nippon\s+india|tata|sundaram|idfc|uti|dsp|kotak|hdfc|icici\s+prudential|sbi|"
    r"aditya\s+birla\s+sun\s+life|birla\s+sun\s+life|mirae\s+asset|franklin\s+templeton|l&t|lt|"
    r"quant|motilal\s+oswal|axis|canara\s+robeco|edelweiss|invesco|baroda|union|pgim\s+india|"
    r"hsbc|mahindra\s+manulife|bandhan|deutsche)?\s*mutual\s+fund(?:\s*\(nimf\))?$",
    re.I,
)

# SEBI / legal boilerplate to ignore when extracting scheme names
BOILERPLATE = [
    r"offer of units", r"new fund offer", r"\bnfo\b", r"statement of additional",
    r"scheme information document", r"^\s*\(?\s*sid\s*\)?\s*$", r"key information",
    r"investor service", r"distributor", r"table of contents", r"trustee company",
    r"asset management", r"name of (the )?mutual fund", r"name of asset management",
    r"stock exchange", r"continuous offer", r"statutory", r"date of this document",
    r"this document contains", r"please refer", r"opens on", r"closes on",
    r"re-opens on", r"the particulars of the scheme", r"securities and exchange",
    r"registered address", r"corporate office", r"website", r"this product is suitable",
    r"suitable for investors", r"who are seeking", r"^\s*page\s*no\.?", r"^\s*index\s*$",
    r"highlights of the scheme", r"summary of the scheme", r"standard risk factors",
    r"specific risk factors", r"exchanges? name in this sid", r"granting the aforesaid permission",
    r"scrutinized this sid", r"fulfilling the various criteria", r"mutual fund investor service",
    r"as amended till date", r"due diligence certificate",
    r"\b(open|close|closed|interval)[\s\-]+ended\b", r"\bmaturity of\b",
    r"\bpredominantly investing in\b", r"\bstatutory lock[\s\-]*in\b",
    r"hereinafter referred to", r"permission to.*use", r"dated.*permission",
    r"empanelled brokers", r"as the scheme is", r"does not warrant",
    r"^\s*\(.*(scheme|fund).*\)\s*$", r"^\s*\(?\s*a\s+(close|open|closed)\b",
    r"^\s*\(?\s*\d+\s*days?\s*close\b"
]

SCHEME_KEYWORDS = re.compile(
    r"\b(fund|etf|fof|series|fmp|index|plan|growth|hybrid|debt|equity|liquid)\b", re.I
)


def clean_text(t: str) -> str:
    """Strip leading page numbers, repeated headers, and normalize whitespace."""
    # Strip leading digits, hyphens, periods (e.g. '1    Axis...')
    t = re.sub(r"^\s*[\d\.\-\_]+\s*", "", t)
    # Strip leading SID / KIM / KIM-SID labels
    t = re.sub(r"^(?:scheme\s+information\s+document|\(?sid\)?|key\s+information\s+memorandum|\(?kim\)?|[\-\–\:\(\)\s])+", "", t, flags=re.I)
    # Strip trailing page numbers or trailing digits
    t = re.sub(r"\s+\d+\s*$", "", t)
    # Collapse multiple spaces to single space
    t = re.sub(r"\s+", " ", t).strip()
    # Deduplicate doubled phrases (e.g. 'ABC Fund ABC Fund' -> 'ABC Fund')
    t = re.sub(r"\b(.+?)\s+\1\b", r"\1", t)
    return t.strip(" ()[]{}—–-_")


def is_boilerplate(line: str) -> bool:
    """Return True if line is a disclaimer, header, category description, or legal boilerplate."""
    line_clean = line.strip()
    if len(line_clean) < 4:
        return True
    if AMC_ONLY.match(line_clean):
        return True
    return any(re.search(b, line_clean, re.I) for b in BOILERPLATE)


def extract_scheme_name(path: Path) -> str | None:
    """Extract the primary scheme name from the first 1-3 pages of the SID PDF."""
    try:
        reader = PdfReader(path)
    except Exception:
        return None

    num_pages = min(3, len(reader.pages))
    candidates = []

    for page_idx in range(num_pages):
        try:
            text = reader.pages[page_idx].extract_text() or ""
        except Exception:
            continue

        raw_lines = [l.strip() for l in text.splitlines() if l.strip()]

        # Priority 1: Check lines immediately BEFORE or AFTER 'SCHEME INFORMATION DOCUMENT'
        for i, raw_l in enumerate(raw_lines):
            if re.search(r"scheme\s+information\s+document", raw_l, re.I):
                # Check lines BEFORE (in reverse order, closest first)
                for j in range(i - 1, max(-1, i - 4), -1):
                    prev_l = raw_lines[j]
                    c_prev = clean_text(prev_l)
                    if not is_boilerplate(prev_l) and not is_boilerplate(c_prev) and SCHEME_KEYWORDS.search(c_prev):
                        # Multi-line continuation: check if line directly above is part of title
                        if j > 0 and len(raw_lines[j - 1]) > 3 and not is_boilerplate(raw_lines[j - 1]):
                            combined = clean_text(raw_lines[j - 1] + " " + prev_l)
                            if len(combined) <= 70 and not is_boilerplate(combined):
                                return combined
                        return c_prev

                # Check lines AFTER (closest first)
                for next_l in raw_lines[i + 1 : i + 6]:
                    c_next = clean_text(next_l)
                    if not is_boilerplate(next_l) and not is_boilerplate(c_next) and SCHEME_KEYWORDS.search(c_next):
                        return c_next

        # Priority 2: Any non-boilerplate line containing a scheme keyword
        lines = [clean_text(l) for l in raw_lines if not is_boilerplate(l)]
        for l in lines:
            if not is_boilerplate(l) and SCHEME_KEYWORDS.search(l):
                return l
            elif page_idx == 0 and len(candidates) < 2:
                candidates.append(l)

    return candidates[0] if candidates else None


def sanitize_filename(name: str) -> str:
    """Make string safe for Windows filesystem while preserving readable words."""
    # Remove filesystem illegal characters: \ / : * ? " < > |
    safe = re.sub(r'[\\/*?:"<>|]', "", name)
    safe = re.sub(r"\s+", " ", safe).strip()
    # Clean cut at word boundary if longer than 70 characters
    if len(safe) > 70:
        safe = safe[:70].rsplit(" ", 1)[0]
    return safe.strip(" ()[]{}—–-_")


def main():
    parser = argparse.ArgumentParser(description="Rename AMFI SID PDFs with extracted Scheme Names.")
    parser.add_argument("--dry-run", action="store_true", help="Preview proposed renames without changing files.")
    parser.add_argument("--revert", action="store_true", help="Revert all files back to pure {doc_id}.pdf.")
    parser.add_argument("--dir", type=str, default=str(OUT_DIR), help="Path to amfi_sids directory.")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    files = sorted(target_dir.glob("*.pdf"))
    total = len(files)
    print(f"Found {total} PDF files in '{target_dir}'.")
    if args.dry_run:
        print("[DRY-RUN MODE] No files will be modified.\n")
    elif args.revert:
        print("[REVERT MODE] Reverting all files to clean numeric IDs ({doc_id}.pdf)...\n")

    renamed_count = 0
    unchanged_count = 0
    reverted_count = 0
    t0 = time.time()

    for i, p in enumerate(files, 1):
        # Extract the original numeric doc_id preserved at the end of the filename
        m = re.search(r"(\d+)\.pdf$", p.name)
        if not m:
            continue
        doc_id = m.group(1)

        # Revert mode: change back to {doc_id}.pdf
        if args.revert:
            target_name = f"{doc_id}.pdf"
            if p.name != target_name:
                if not args.dry_run:
                    p.rename(p.with_name(target_name))
                reverted_count += 1
            else:
                unchanged_count += 1
            continue

        # Normal rename mode:
        scheme_name = extract_scheme_name(p)
        if scheme_name:
            safe = sanitize_filename(scheme_name)
            target_name = f"{safe}_{doc_id}.pdf"
        else:
            # Fall back to clean numeric id if no scheme name found
            target_name = f"{doc_id}.pdf"

        if p.name == target_name:
            unchanged_count += 1
        else:
            if not args.dry_run:
                dest = p.with_name(target_name)
                if dest.exists() and dest != p:
                    target_name = f"{safe}_{doc_id}_dup.pdf"
                    dest = p.with_name(target_name)
                p.rename(dest)
            renamed_count += 1

        if i % 250 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            if args.revert:
                print(f"[{i}/{total}] Reverted: {reverted_count} | Unchanged: {unchanged_count} ({rate:.1f} files/s)")
            else:
                print(f"[{i}/{total}] Renamed: {renamed_count} | Unchanged: {unchanged_count} ({rate:.1f} files/s)")

    elapsed = time.time() - t0
    print("\n" + "=" * 50)
    if args.revert:
        print(f"Done. Reverted {reverted_count} files in {elapsed:.1f}s.")
    else:
        print(f"Done. Renamed {renamed_count} files, {unchanged_count} unchanged in {elapsed:.1f}s.")
    print("=" * 50)


if __name__ == "__main__":
    main()
