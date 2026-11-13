#!/usr/bin/env python3
"""Fetch LIVE headlines for a ticker (Google News RSS, free, no key) and
write the English ones to a JSON list for sentiment scoring.

Shared by the fleet's sentiment runners so they always score the day's
actual coverage of the actual ticker — never a stored sample. Arabic
headlines are counted and logged but not scored (the sentiment models
are English-only).

Usage: python scripts/fetch_headlines.py --ticker POUL.CA --out headlines.json
"""
import argparse, json, re, sys, urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

UA = {"User-Agent": "Moos Labs research (moos@sherbeeny.com)"}
ALIASES = {
    "POUL.CA": {"ar": "القاهرة للدواجن", "en": "Cairo Poultry"},
    "COMI.CA": {"ar": "البنك التجاري الدولي", "en": "Commercial International Bank Egypt"},
    "ETEL.CA": {"ar": "المصرية للاتصالات", "en": "Telecom Egypt"},
    "HRHO.CA": {"ar": "إي إف جي هيرميس", "en": "EFG Hermes"},
    "TMGH.CA": {"ar": "طلعت مصطفى", "en": "Talaat Moustafa Group"},
    "SWDY.CA": {"ar": "السويدي إلكتريك", "en": "Elsewedy Electric"},
    "ESRS.CA": {"ar": "حديد عز", "en": "Ezz Steel"},
    "ABUK.CA": {"ar": "أبو قير للأسمدة", "en": "Abu Qir Fertilizers"},
    "AMOC.CA": {"ar": "الإسكندرية للزيوت المعدنية", "en": "AMOC Egypt"},
    "EKHO.CA": {"ar": "القابضة المصرية الكويتية", "en": "Egypt Kuwait Holding"},
}


def is_arabic(t):
    return any("\u0600" <= ch <= "\u06FF" for ch in t or "")


def gnews(query, lang, country, days):
    q = f"{query} when:{days}d"
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(q)
           + f"&hl={lang}&gl={country}&ceid={country}:{lang}")
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    out = []
    for it in ET.fromstring(r.content).iter("item"):
        out.append({"title": (it.findtext("title") or "").strip(),
                    "date": it.findtext("pubDate") or ""})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", default="headlines.json")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--max", type=int, default=40)
    args = ap.parse_args()

    ticker = args.ticker.upper()
    alias = ALIASES.get(ticker, {})
    queries = []
    if ticker.endswith(".CA"):
        if alias.get("en"):
            queries.append((alias["en"], "en", "US"))
        queries.append((ticker + " stock", "en", "US"))
        queries.append((ticker, "en", "US"))
    else:
        bare = ticker.split(".")[0]
        queries.append((f"{bare} stock", "en", "US"))
        queries.append((f"{bare} earnings", "en", "US"))

    days = args.days
    items, seen = [], set()
    for _ in range(2):  # widen the window once if English coverage is thin
        for q, lang, country in queries:
            try:
                got = gnews(q, lang, country, days)
            except Exception as e:
                print(f"WARN gnews failed ({q}): {e}", file=sys.stderr)
                continue
            for it in got:
                key = re.sub(r"\s+", " ", re.sub(r" - [^-]+$", "", it["title"])).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    items.append(it)
        en = [i for i in items if not is_arabic(i["title"])]
        if len(en) >= 3 or days >= 90:
            break
        days = 90

    def sortkey(it):
        try:
            return datetime.strptime(it["date"], "%a, %d %b %Y %H:%M:%S %Z")
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    items.sort(key=sortkey, reverse=True)
    en = [i["title"] for i in items if not is_arabic(i["title"])][:args.max]
    ar = sum(1 for i in items if is_arabic(i["title"]))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(en, f, ensure_ascii=False, indent=1)
    print(f"headlines: {len(en)} English scored-input, {ar} Arabic skipped, "
          f"window {days}d, ticker {ticker}")
    if not en:
        print("ERROR: no English headlines found for this ticker", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
