"""Refresh data/holidays/holidays.json from the NSE holiday API.

NSE publishes its trading calendar as JSON (segment keys: CM cash, FO derivatives, COM
commodity derivatives on NSE, ...). Dhan has no holiday API; its public page
https://dhan.co/market-holiday/ shows the MCX morning/evening split, which is kept by hand
in the MCX block of holidays.json and preserved by this script.

Usage:
    python scripts/fetch_holidays.py            # refresh NSE from the API, keep MCX
    python scripts/fetch_holidays.py --print    # show the merged calendar
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

NSE_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HOLIDAYS_FILE = Path(__file__).resolve().parents[1] / "data" / "holidays" / "holidays.json"


def fetch_nse() -> dict[str, list[dict]]:
    r = httpx.get(
        NSE_URL,
        headers={
            "User-Agent": UA,
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        },
        timeout=30,
        follow_redirects=True,
    )
    r.raise_for_status()
    return r.json()


def to_iso(d: str) -> str:
    return datetime.strptime(d, "%d-%b-%Y").date().isoformat()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=HOLIDAYS_FILE)
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()

    current = json.loads(args.file.read_text()) if args.file.exists() else {}
    if not args.print:
        raw = fetch_nse()
        cm = raw.get("CM") or raw.get("CBM") or []
        fo = raw.get("FO") or cm
        nse_days = sorted(
            {to_iso(r["tradingDate"]) for r in cm} | {to_iso(r["tradingDate"]) for r in fo}
        )
        names = {to_iso(r["tradingDate"]): r["description"] for r in cm + fo}
        nse_block = current.get("NSE", {})
        nse_block["holidays"] = nse_days
        nse_block["names"] = names
        nse_block["source"] = NSE_URL
        nse_block["fetched_at"] = datetime.now().isoformat(timespec="seconds")
        current["NSE"] = nse_block
        current.setdefault("MCX", {"holidays": [], "special_sessions": {}})
        args.file.write_text(json.dumps(current, indent=2) + "\n")
        print(f"NSE: {len(nse_days)} holidays written to {args.file}")
    for ex in ("NSE", "MCX"):
        blk = current.get(ex, {})
        print(
            f"\n{ex}: {len(blk.get('holidays', []))} full holidays, "
            f"{len(blk.get('special_sessions', {}))} special sessions"
        )
        if args.print:
            for d in blk.get("holidays", []):
                print(f"  {d}  {blk.get('names', {}).get(d, '')}")
            for d, s in sorted(blk.get("special_sessions", {}).items()):
                print(f"  {d}  special {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
