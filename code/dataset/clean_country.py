#!/usr/bin/env python3
"""Clean the dirty `country` field of the fern split CSVs.

The raw column mixes ISO-2 codes ("CN"), iNaturalist place strings
("Dali Bai, CN-YN, CN"), Chinese addresses ("台灣台北"), Russian locality
strings, English names ("Pingxiang, Jiangxi, China") and empty values
(1,468 distinct values in train alone) -- unusable for any geographic
split. This script ADDS a `country_clean` column (raw data untouched):

  1. empty / "ZZ" (iNaturalist unknown)            -> UNK
  2. bare ISO-2 code from a known list             -> itself
  3. Chinese text 中國/中国/台灣/臺灣                -> CN / TW
  4. full country names / Russian Россия           -> ISO-2
  5. any ISO-2 token, incl. ISO-3166-2 "CN-YN"     -> that country
  6. anything else                                 -> UNK (logged)

    python clean_country.py                 # clean all three splits in place
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

SPLITS = Path(__file__).resolve().parents[2] / "data/splits"

# ISO-2 codes seen or plausibly seen in this dataset (ferns are global,
# but the corpus is Asia-heavy with a European/African/American tail).
ISO2 = {
    "CN", "TW", "HK", "MO", "JP", "KR", "KP", "MN", "TH", "VN", "KH", "LA",
    "MM", "MY", "SG", "BN", "ID", "PH", "IN", "NP", "BT", "BD", "PK", "LK",
    "RU", "KZ", "GE", "AM", "AZ", "TR", "IR", "IL", "SA", "AF", "US", "CA",
    "MX", "BR", "AR", "CL", "CO", "PE", "EC", "BO", "PY", "UY", "VE", "GY",
    "SR", "CR", "PA", "CU", "JM", "HT", "DO", "PR", "GB", "IE", "FR", "DE",
    "PL", "CZ", "SK", "HU", "AT", "CH", "IT", "ES", "PT", "NL", "BE", "LU",
    "DK", "SE", "FI", "NO", "IS", "EE", "LV", "LT", "BY", "UA", "MD", "RO",
    "BG", "GR", "HR", "SI", "RS", "BA", "MK", "AL", "ME", "CY", "MT", "ZA",
    "MG", "CV", "CM", "KE", "TZ", "UG", "ET", "RW", "BI", "CD", "CG", "GA",
    "GQ", "NG", "GH", "CI", "SN", "GM", "ML", "BF", "NE", "TD", "SD", "MZ",
    "ZW", "ZM", "BW", "NA", "AO", "AU", "NZ", "PG", "FJ", "SB", "VU", "NC",
    "PW", "FM", "TL", "LR", "GF", "MQ", "GP", "RE", "YT", "PF", "GU", "VI",
}

# Full names / native spellings -> ISO-2. Matched case-insensitively as
# substrings, longest key first.
NAMES = {
    "china": "CN", "中国大陆": "CN", "中华人民共和国": "CN",
    "taiwan": "TW", "台灣": "TW", "臺灣": "TW", "台湾": "TW",
    "hong kong": "HK", "香港": "HK",
    "japan": "JP", "日本": "JP", "nippon": "JP",
    "korea": "KR", "韓國": "KR", "한국": "KR",
    "vietnam": "VN", "việt nam": "VN", "越南": "VN",
    "thailand": "TH", "ไทย": "TH", "泰国": "TH",
    "india": "IN", "भारत": "IN", "印度": "IN",
    "nepal": "NP", "नेपाल": "NP", "尼泊尔": "NP",
    "indonesia": "ID", "印度尼西亚": "ID",
    "malaysia": "MY", "马来西亚": "MY",
    "philippines": "PH", "pilipinas": "PH", "菲律宾": "PH",
    "laos": "LA", "老挝": "LA",
    "myanmar": "MM", "burma": "MM", "缅甸": "MM",
    "cambodia": "KH", "柬埔寨": "KH",
    "russia": "RU", "россия": "RU", "俄罗斯": "RU",
    "united states": "US", "usa": "US", "u.s.a.": "US", "美国": "US",
    "united kingdom": "GB", "uk": "GB", "britain": "GB", "england": "GB",
    "德国": "DE", "germany": "DE", "deutschland": "DE",
    "france": "FR", "法国": "FR",
    "poland": "PL", "polska": "PL", "波兰": "PL",
    "spain": "ES", "españa": "ES", "西班牙": "ES",
    "italy": "IT", "italia": "IT", "意大利": "IT",
    "netherlands": "NL", "holland": "NL", "荷兰": "NL",
    "sweden": "SE", "sverige": "SE", "瑞典": "SE",
    "south africa": "ZA", "南非": "ZA",
    "madagascar": "MG", "马达加斯加": "MG",
    "cabo verde": "CV", "cape verde": "CV", "佛得角": "CV",
    "mexico": "MX", "méxico": "MX", "墨西哥": "MX",
    "brazil": "BR", "brasil": "BR", "巴西": "BR",
    "australia": "AU", "澳大利亚": "AU",
    "new zealand": "NZ", "新西兰": "NZ",
    "puerto rico": "PR", "switzerland": "CH", "schweiz": "CH", "suisse": "CH",
    "portugal": "PT", "canada": "CA", "slovensko": "SK", "slovakia": "SK",
    "lietuva": "LT", "lithuania": "LT", "taiwán": "TW", "таиланд": "TH",
    "대한민국": "KR", "guyane": "GF", "martinique": "MQ",
}

# Locality strings that carry no country token at all: known region names.
# Taiwan counties/cities (traditional + simplified) and Malaysian states.
_LOCAL_RE = re.compile(
    r"宜蘭|宜兰|嘉義|嘉义|桃園|桃园|新竹|花蓮|花莲|南投|雲林|云林|屏東|屏东|"
    r"高雄|臺南|台南|臺北|台北|新北|苗栗|彰化|基隆|澎湖|金門|金门|連江|连江|"
    r"臺東|台東|玉山|阿里山|太平山|合歡山|雪山")
_MY_LOCAL_RE = re.compile(r"砂拉越|美里|怡保|吉隆坡|槟城|柔佛|马六甲|沙巴|亞庇|亚庇")

# ISO-2 token, optionally with an ISO-3166-2 suffix: CN, CN-YN, TW-NT-TP
_ISO2_RE = re.compile(r"\b([A-Z]{2})(?:-[A-Z0-9]{1,3})?\b")
_CN_RE = re.compile(r"中國|中国|中华人民共和国")
_TW_RE = re.compile(r"台灣|臺灣|台湾")


def clean(raw: str) -> str:
    v = (raw or "").strip()
    if not v or v.upper() == "ZZ":
        return "UNK"
    # bare code
    if len(v) == 2 and v.upper() in ISO2:
        return v.upper()
    # Chinese text first (a Chinese string may also contain ASCII codes)
    if _CN_RE.search(v):
        return "CN"
    if _TW_RE.search(v):
        return "TW"
    # locality strings without any country token
    if _LOCAL_RE.search(v):
        return "TW"
    if _MY_LOCAL_RE.search(v):
        return "MY"
    # full / native names, longest key first
    low = v.lower()
    for name in sorted(NAMES, key=len, reverse=True):
        if name in low:
            return NAMES[name]
    # any ISO-2 token; take the LAST match (locality strings end with the
    # country: "Dali Bai, CN-YN, CN" -> CN, "New Taipei, TW-NT-TP, TW" -> TW)
    toks = _ISO2_RE.findall(v.upper())
    toks = [t for t in toks if t in ISO2]
    if toks:
        return toks[-1]
    return "UNK"


def main() -> int:
    total = Counter()
    unresolved = Counter()
    for name in ("train", "val", "test"):
        p = SPLITS / f"{name}.csv"
        with p.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            fields = reader.fieldnames or []
        if "country_clean" in fields:
            fields.remove("country_clean")
        fields = fields + ["country_clean"]
        for r in rows:
            c = clean(r.get("country", ""))
            r["country_clean"] = c
            total[c] += 1
            if c == "UNK":
                unresolved[(r.get("country") or "").strip()[:60]] += 1
        with p.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        cc = Counter(r["country_clean"] for r in rows)
        print(f"{name}: {len(rows)} rows -> {len(cc)} countries; "
              f"UNK={cc.get('UNK', 0)}")
    print("\n=== country_clean distribution (all splits) ===")
    for k, v in total.most_common(30):
        print(f"  {k}: {v}")
    if unresolved:
        print(f"\n=== unresolved -> UNK ({sum(unresolved.values())} rows, "
              f"{len(unresolved)} distinct raw values) ===")
        for k, v in unresolved.most_common(25):
            print(f"  {k!r}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
