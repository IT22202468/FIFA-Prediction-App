"""
Phase 5 — Full FIFA World Cup 2026 Tournament Simulation
=========================================================
Simulates the complete 2026 World Cup (group stage + all knockout rounds)
N times and reports the probability of each team reaching every round.

Usage
-----
    python src/simulate.py                       # 10,000 sims, print table
    python src/simulate.py --sims 50000          # more precision
    python src/simulate.py --seed 42             # reproducible
    python src/simulate.py --out data/sim_results.json   # save JSON

Python API
----------
    from src.simulate import run
    results = run(n_sims=10_000, seed=42)
    # results is a dict: {team: {"R32": p, "R16": p, ..., "Champion": p}}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SRC_DIR  = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import load_params, _dc_joint_pmf  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Official 2026 World Cup group assignments (FIFA draw, December 2023)
# ─────────────────────────────────────────────────────────────────────────────
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",      "South Korea",  "Czech Republic",        "South Africa"],
    "B": ["Canada",      "Qatar",        "Switzerland",           "Bosnia and Herzegovina"],
    "C": ["Brazil",      "Morocco",      "Haiti",                 "Scotland"],
    "D": ["United States", "Australia",  "Paraguay",              "Turkey"],
    "E": ["Germany",     "Côte d'Ivoire", "Ecuador",             "Curaçao"],
    "F": ["Netherlands", "Japan",        "Sweden",                "Tunisia"],
    "G": ["Belgium",     "Egypt",        "Iran",                  "New Zealand"],
    "H": ["Spain",       "Cabo Verde",   "Saudi Arabia",          "Uruguay"],
    "I": ["France",      "Senegal",      "Iraq",                  "Norway"],
    "J": ["Argentina",   "Algeria",      "Austria",               "Jordan"],
    "K": ["Portugal",    "DR Congo",     "Uzbekistan",            "Colombia"],
    "L": ["England",     "Croatia",      "Ghana",                 "Panama"],
}

# Name aliases: dataset variants → canonical name in GROUPS above
TEAM_ALIASES: dict[str, str] = {
    "Cape Verde Islands": "Cabo Verde",
    "Czechia":            "Czech Republic",
    "Turkey":             "Turkey",
}

# Reverse map: team → group letter
TEAM_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}

# ─────────────────────────────────────────────────────────────────────────────
# Bracket: R32 fixed matchups (both opponents are 1st or 2nd place)
# ─────────────────────────────────────────────────────────────────────────────
# Format: (match_id, slot1, slot2)
# "1X" = winner of group X, "2X" = runner-up
R32_FIXED = [
    (73, "2A", "2B"),
    (75, "1F", "2C"),
    (76, "1C", "2F"),
    (78, "2E", "2I"),
    (83, "2K", "2L"),
    (84, "1H", "2J"),
    (86, "1J", "2H"),
    (88, "2D", "2G"),
]

# R32 matches where one slot is a 3rd-place qualifier
# {match_id: confirmed_slot (the group winner or runner-up)}
# The other opponent comes from the 495-combo lookup
R32_THIRD: dict[int, str] = {
    74: "1E",
    77: "1I",
    79: "1A",
    80: "1L",
    81: "1D",
    82: "1G",
    85: "1B",
    87: "1K",
}

# The 8 columns of the combo assignment string map to these match IDs
# (column order: 1A vs, 1B vs, 1D vs, 1E vs, 1G vs, 1I vs, 1K vs, 1L vs)
_THIRD_COL_TO_MATCH = [79, 85, 81, 74, 82, 77, 87, 80]

# R16, QF, SF bracket tree  (match_id, feeder1, feeder2)
R16 = [(89,74,77),(90,73,75),(91,76,78),(92,79,80),
       (93,83,84),(94,81,82),(95,86,88),(96,85,87)]
QF  = [(97,89,90),(98,93,94),(99,91,92),(100,95,96)]
SF  = [(101,97,98),(102,99,100)]
THIRD_PLACE_MATCH = (103, 101, 102)  # losers of SF
FINAL             = (104, 101, 102)  # winners of SF

KNOCKOUT_ORDER = R16 + QF + SF + [THIRD_PLACE_MATCH, FINAL]

# Rounds, in progression order
ROUND_NAMES = ["R32", "R16", "QF", "SF", "Final", "Champion"]

# ─────────────────────────────────────────────────────────────────────────────
# 495 combinations table (FIFA Annex C — official bracket seeding)
# Key = sorted 8-char string of group letters whose 3rd-place teams qualify
# Value = 8-char assignment string: position i → 3rd-place team from group
#   value[i], matching _THIRD_COL_TO_MATCH[i]
# ─────────────────────────────────────────────────────────────────────────────
_COMBO_DICT: dict[str, str] = {
    "ABCDEFGH": "HGBCAFDE",
    "ABCDEFGI": "CGBDAFEI",
    "ABCDEFGJ": "CGBDAFEJ",
    "ABCDEFGK": "CGBDAFEK",
    "ABCDEFGL": "CGBDAFLE",
    "ABCDEFHI": "HEBCAFDI",
    "ABCDEFHJ": "HJBCAFDE",
    "ABCDEFHK": "HEBCAFDK",
    "ABCDEFHL": "HFBCADLE",
    "ABCDEFIJ": "CJBDAFEI",
    "ABCDEFIK": "CEBDAFIK",
    "ABCDEFIL": "CEBDAFLI",
    "ABCDEFJK": "CJBDAFEK",
    "ABCDEFJL": "CJBDAFLE",
    "ABCDEFKL": "CEBDAFLK",
    "ABCDEGHI": "HGBCADEI",
    "ABCDEGHJ": "HGBCADEJ",
    "ABCDEGHK": "HGBCADEK",
    "ABCDEGHL": "HGBCADLE",
    "ABCDEGIJ": "EGBCADIJ",
    "ABCDEGIK": "EGBCADIK",
    "ABCDEGIL": "EGBCADLI",
    "ABCDEGJK": "EGBCADJK",
    "ABCDEGJL": "EGBCADLJ",
    "ABCDEGKL": "EGBCADLK",
    "ABCDEHIJ": "HJBCADEI",
    "ABCDEHIK": "HEBCADIK",
    "ABCDEHIL": "HEBCADLI",
    "ABCDEHJK": "HJBCADEK",
    "ABCDEHJL": "HJBCADLE",
    "ABCDEHKL": "HEBCADLK",
    "ABCDEIJK": "EJBCADIK",
    "ABCDEIJL": "EJBCADLI",
    "ABCDEIKL": "EIBCADLK",
    "ABCDEJKL": "EJBCADLK",
    "ABCDFGHI": "HGBCAFDI",
    "ABCDFGHJ": "HGBCAFDJ",
    "ABCDFGHK": "HGBCAFDK",
    "ABCDFGHL": "CGBDAFLH",
    "ABCDFGIJ": "CGBDAFIJ",
    "ABCDFGIK": "CGBDAFIK",
    "ABCDFGIL": "CGBDAFLI",
    "ABCDFGJK": "CGBDAFJK",
    "ABCDFGJL": "CGBDAFLJ",
    "ABCDFGKL": "CGBDAFLK",
    "ABCDFHIJ": "HJBCAFDI",
    "ABCDFHIK": "HFBCADIK",
    "ABCDFHIL": "HFBCADLI",
    "ABCDFHJK": "HJBCAFDK",
    "ABCDFHJL": "CJBDAFLH",
    "ABCDFHKL": "HFBCADLK",
    "ABCDFIJK": "CJBDAFIK",
    "ABCDFIJL": "CJBDAFLI",
    "ABCDFIKL": "CIBDAFLK",
    "ABCDFJKL": "CJBDAFLK",
    "ABCDGHIJ": "HGBCADIJ",
    "ABCDGHIK": "HGBCADIK",
    "ABCDGHIL": "HGBCADLI",
    "ABCDGHJK": "HGBCADJK",
    "ABCDGHJL": "HGBCADLJ",
    "ABCDGHKL": "HGBCADLK",
    "ABCDGIJK": "CJBDAGIK",
    "ABCDGIJL": "CJBDAGLI",
    "ABCDGIKL": "IGBCADLK",
    "ABCDGJKL": "CJBDAGLK",
    "ABCDHIJK": "HJBCADIK",
    "ABCDHIJL": "HJBCADLI",
    "ABCDHIKL": "HIBCADLK",
    "ABCDHJKL": "HJBCADLK",
    "ABCDIJKL": "IJBCADLK",
    "ABCEFGHI": "HGBCAFEI",
    "ABCEFGHJ": "HGBCAFEJ",
    "ABCEFGHK": "HGBCAFEK",
    "ABCEFGHL": "HGBCAFLE",
    "ABCEFGIJ": "EGBCAFIJ",
    "ABCEFGIK": "EGBCAFIK",
    "ABCEFGIL": "EGBCAFLI",
    "ABCEFGJK": "EGBCAFJK",
    "ABCEFGJL": "EGBCAFLJ",
    "ABCEFGKL": "EGBCAFLK",
    "ABCEFHIJ": "HJBCAFEI",
    "ABCEFHIK": "HEBCAFIK",
    "ABCEFHIL": "HEBCAFLI",
    "ABCEFHJK": "HJBCAFEK",
    "ABCEFHJL": "HJBCAFLE",
    "ABCEFHKL": "HEBCAFLK",
    "ABCEFIJK": "EJBCAFIK",
    "ABCEFIJL": "EJBCAFLI",
    "ABCEFIKL": "EIBCAFLK",
    "ABCEFJKL": "EJBCAFLK",
    "ABCEGHIJ": "HJBCAGEI",
    "ABCEGHIK": "EGBCAHIK",
    "ABCEGHIL": "EGBCAHLI",
    "ABCEGHJK": "HJBCAGEK",
    "ABCEGHJL": "HJBCAGLE",
    "ABCEGHKL": "EGBCAHLK",
    "ABCEGIJK": "EJBCAGIK",
    "ABCEGIJL": "EJBCAGLI",
    "ABCEGIKL": "EGBAICLK",
    "ABCEGJKL": "EJBCAGLK",
    "ABCEHIJK": "EJBCAHIK",
    "ABCEHIJL": "EJBCAHLI",
    "ABCEHIKL": "EIBCAHLK",
    "ABCEHJKL": "EJBCAHLK",
    "ABCEIJKL": "EJBAICLK",
    "ABCFGHIJ": "HGBCAFIJ",
    "ABCFGHIK": "HGBCAFIK",
    "ABCFGHIL": "HGBCAFLI",
    "ABCFGHJK": "HGBCAFJK",
    "ABCFGHJL": "HGBCAFLJ",
    "ABCFGHKL": "HGBCAFLK",
    "ABCFGIJK": "CJBFAGIK",
    "ABCFGIJL": "CJBFAGLI",
    "ABCFGIKL": "IGBCAFLK",
    "ABCFGJKL": "CJBFAGLK",
    "ABCFHIJK": "HJBCAFIK",
    "ABCFHIJL": "HJBCAFLI",
    "ABCFHIKL": "HIBCAFLK",
    "ABCFHJKL": "HJBCAFLK",
    "ABCFIJKL": "IJBCAFLK",
    "ABCGHIJK": "HJBCAGIK",
    "ABCGHIJL": "HJBCAGLI",
    "ABCGHIKL": "IGBCAHLK",
    "ABCGHJKL": "HJBCAGLK",
    "ABCGIJKL": "IJBCAGLK",
    "ABCHIJKL": "IJBCAHLK",
    "ABDEFGHI": "HGBDAFEI",
    "ABDEFGHJ": "HGBDAFEJ",
    "ABDEFGHK": "HGBDAFEK",
    "ABDEFGHL": "HGBDAFLE",
    "ABDEFGIJ": "EGBDAFIJ",
    "ABDEFGIK": "EGBDAFIK",
    "ABDEFGIL": "EGBDAFLI",
    "ABDEFGJK": "EGBDAFJK",
    "ABDEFGJL": "EGBDAFLJ",
    "ABDEFGKL": "EGBDAFLK",
    "ABDEFHIJ": "HJBDAFEI",
    "ABDEFHIK": "HEBDAFIK",
    "ABDEFHIL": "HEBDAFLI",
    "ABDEFHJK": "HJBDAFEK",
    "ABDEFHJL": "HJBDAFLE",
    "ABDEFHKL": "HEBDAFLK",
    "ABDEFIJK": "EJBDAFIK",
    "ABDEFIJL": "EJBDAFLI",
    "ABDEFIKL": "EIBDAFLK",
    "ABDEFJKL": "EJBDAFLK",
    "ABDEGHIJ": "HJBDAGEI",
    "ABDEGHIK": "EGBDAHIK",
    "ABDEGHIL": "EGBDAHLI",
    "ABDEGHJK": "HJBDAGEK",
    "ABDEGHJL": "HJBDAGLE",
    "ABDEGHKL": "EGBDAHLK",
    "ABDEGIJK": "EJBDAGIK",
    "ABDEGIJL": "EJBDAGLI",
    "ABDEGIKL": "EGBAIDLK",
    "ABDEGJKL": "EJBDAGLK",
    "ABDEHIJK": "EJBDAHIK",
    "ABDEHIJL": "EJBDAHLI",
    "ABDEHIKL": "EIBDAHLK",
    "ABDEHJKL": "EJBDAHLK",
    "ABDEIJKL": "EJBAIDLK",
    "ABDFGHIJ": "HGBDAFIJ",
    "ABDFGHIK": "HGBDAFIK",
    "ABDFGHIL": "HGBDAFLI",
    "ABDFGHJK": "HGBDAFJK",
    "ABDFGHJL": "HGBDAFLJ",
    "ABDFGHKL": "HGBDAFLK",
    "ABDFGIJK": "FJBDAGIK",
    "ABDFGIJL": "FJBDAGLI",
    "ABDFGIKL": "IGBDAFLK",
    "ABDFGJKL": "FJBDAGLK",
    "ABDFHIJK": "HJBDAFIK",
    "ABDFHIJL": "HJBDAFLI",
    "ABDFHIKL": "HIBDAFLK",
    "ABDFHJKL": "HJBDAFLK",
    "ABDFIJKL": "IJBDAFLK",
    "ABDGHIJK": "HJBDAGIK",
    "ABDGHIJL": "HJBDAGLI",
    "ABDGHIKL": "IGBDAHLK",
    "ABDGHJKL": "HJBDAGLK",
    "ABDGIJKL": "IJBDAGLK",
    "ABDHIJKL": "IJBDAHLK",
    "ABEFGHIJ": "HJBFAGEI",
    "ABEFGHIK": "EGBFAHIK",
    "ABEFGHIL": "EGBFAHLI",
    "ABEFGHJK": "HJBFAGEK",
    "ABEFGHJL": "HJBFAGLE",
    "ABEFGHKL": "EGBFAHLK",
    "ABEFGIJK": "EJBFAGIK",
    "ABEFGIJL": "EJBFAGLI",
    "ABEFGIKL": "EGBAIFLK",
    "ABEFGJKL": "EJBFAGLK",
    "ABEFHIJK": "EJBFAHIK",
    "ABEFHIJL": "EJBFAHLI",
    "ABEFHIKL": "EIBFAHLK",
    "ABEFHJKL": "EJBFAHLK",
    "ABEFIJKL": "EJBAIFLK",
    "ABEGHIJK": "EJBAHGIK",
    "ABEGHIJL": "EJBAHGLI",
    "ABEGHIKL": "EGBAIHLK",
    "ABEGHJKL": "EJBAHGLK",
    "ABEGIJKL": "EJBAIGLK",
    "ABEHIJKL": "EJBAIHLK",
    "ABFGHIJK": "HJBFAGIK",
    "ABFGHIJL": "HJBFAGLI",
    "ABFGHIKL": "HGBAIFLK",
    "ABFGHJKL": "HJBFAGLK",
    "ABFGIJKL": "IJBFAGLK",
    "ABFHIJKL": "HJBAIFLK",
    "ABGHIJKL": "HJBAIGLK",
    "ACDEFGHI": "HGECAFDI",
    "ACDEFGHJ": "HGJCAFDE",
    "ACDEFGHK": "HGECAFDK",
    "ACDEFGHL": "HGFCADLE",
    "ACDEFGIJ": "CGJDAFEI",
    "ACDEFGIK": "CGEDAFIK",
    "ACDEFGIL": "CGEDAFLI",
    "ACDEFGJK": "CGJDAFEK",
    "ACDEFGJL": "CGJDAFLE",
    "ACDEFGKL": "CGEDAFLK",
    "ACDEFHIJ": "HJECAFDI",
    "ACDEFHIK": "HEFCADIK",
    "ACDEFHIL": "HEFCADLI",
    "ACDEFHJK": "HJECAFDK",
    "ACDEFHJL": "HJFCADLE",
    "ACDEFHKL": "HEFCADLK",
    "ACDEFIJK": "CJEDAFIK",
    "ACDEFIJL": "CJEDAFLI",
    "ACDEFIKL": "CEIDAFLK",
    "ACDEFJKL": "CJEDAFLK",
    "ACDEGHIJ": "HGJCADEI",
    "ACDEGHIK": "HGECADIK",
    "ACDEGHIL": "HGECADLI",
    "ACDEGHJK": "HGJCADEK",
    "ACDEGHJL": "HGJCADLE",
    "ACDEGHKL": "HGECADLK",
    "ACDEGIJK": "EGJCADIK",
    "ACDEGIJL": "EGJCADLI",
    "ACDEGIKL": "EGICADLK",
    "ACDEGJKL": "EGJCADLK",
    "ACDEHIJK": "HJECADIK",
    "ACDEHIJL": "HJECADLI",
    "ACDEHIKL": "HEICADLK",
    "ACDEHJKL": "HJECADLK",
    "ACDEIJKL": "EJICADLK",
    "ACDFGHIJ": "HGJCAFDI",
    "ACDFGHIK": "HGFCADIK",
    "ACDFGHIL": "HGFCADLI",
    "ACDFGHJK": "HGJCAFDK",
    "ACDFGHJL": "CGJDAFLH",
    "ACDFGHKL": "HGFCADLK",
    "ACDFGIJK": "CGJDAFIK",
    "ACDFGIJL": "CGJDAFLI",
    "ACDFGIKL": "CGIDAFLK",
    "ACDFGJKL": "CGJDAFLK",
    "ACDFHIJK": "HJFCADIK",
    "ACDFHIJL": "HJFCADLI",
    "ACDFHIKL": "HFICADLK",
    "ACDFHJKL": "HJFCADLK",
    "ACDFIJKL": "CJIDAFLK",
    "ACDGHIJK": "HGJCADIK",
    "ACDGHIJL": "HGJCADLI",
    "ACDGHIKL": "HGICADLK",
    "ACDGHJKL": "HGJCADLK",
    "ACDGIJKL": "IGJCADLK",
    "ACDHIJKL": "HJICADLK",
    "ACEFGHIJ": "HGJCAFEI",
    "ACEFGHIK": "HGECAFIK",
    "ACEFGHIL": "HGECAFLI",
    "ACEFGHJK": "HGJCAFEK",
    "ACEFGHJL": "HGJCAFLE",
    "ACEFGHKL": "HGECAFLK",
    "ACEFGIJK": "EGJCAFIK",
    "ACEFGIJL": "EGJCAFLI",
    "ACEFGIKL": "EGICAFLK",
    "ACEFGJKL": "EGJCAFLK",
    "ACEFHIJK": "HJECAFIK",
    "ACEFHIJL": "HJECAFLI",
    "ACEFHIKL": "HEICAFLK",
    "ACEFHJKL": "HJECAFLK",
    "ACEFIJKL": "EJICAFLK",
    "ACEGHIJK": "EGJCAHIK",
    "ACEGHIJL": "EGJCAHLI",
    "ACEGHIKL": "EGICAHLK",
    "ACEGHJKL": "EGJCAHLK",
    "ACEGIJKL": "EJICAGLK",
    "ACEHIJKL": "EJICAHLK",
    "ACFGHIJK": "HGJCAFIK",
    "ACFGHIJL": "HGJCAFLI",
    "ACFGHIKL": "HGICAFLK",
    "ACFGHJKL": "HGJCAFLK",
    "ACFGIJKL": "IGJCAFLK",
    "ACFHIJKL": "HJICAFLK",
    "ACGHIJKL": "HJICAGLK",
    "ADEFGHIJ": "HGJDAFEI",
    "ADEFGHIK": "HGEDAFIK",
    "ADEFGHIL": "HGEDAFLI",
    "ADEFGHJK": "HGJDAFEK",
    "ADEFGHJL": "HGJDAFLE",
    "ADEFGHKL": "HGEDAFLK",
    "ADEFGIJK": "EGJDAFIK",
    "ADEFGIJL": "EGJDAFLI",
    "ADEFGIKL": "EGIDAFLK",
    "ADEFGJKL": "EGJDAFLK",
    "ADEFHIJK": "HJEDAFIK",
    "ADEFHIJL": "HJEDAFLI",
    "ADEFHIKL": "HEIDAFLK",
    "ADEFHJKL": "HJEDAFLK",
    "ADEFIJKL": "EJIDAFLK",
    "ADEGHIJK": "EGJDAHIK",
    "ADEGHIJL": "EGJDAHLI",
    "ADEGHIKL": "EGIDAHLK",
    "ADEGHJKL": "EGJDAHLK",
    "ADEGIJKL": "EJIDAGLK",
    "ADEHIJKL": "EJIDAHLK",
    "ADFGHIJK": "HGJDAFIK",
    "ADFGHIJL": "HGJDAFLI",
    "ADFGHIKL": "HGIDAFLK",
    "ADFGHJKL": "HGJDAFLK",
    "ADFGIJKL": "IGJDAFLK",
    "ADFHIJKL": "HJIDAFLK",
    "ADGHIJKL": "HJIDAGLK",
    "AEFGHIJK": "EGJFAHIK",
    "AEFGHIJL": "EGJFAHLI",
    "AEFGHIKL": "EGIFAHLK",
    "AEFGHJKL": "EGJFAHLK",
    "AEFGIJKL": "EJIFAGLK",
    "AEFHIJKL": "EJIFAHLK",
    "AEGHIJKL": "EJIAHGLK",
    "AFGHIJKL": "HJIFAGLK",
    "BCDEFGHI": "CGBDHFEI",
    "BCDEFGHJ": "HGBCJFDE",
    "BCDEFGHK": "CGBDHFEK",
    "BCDEFGHL": "CGBDHFLE",
    "BCDEFGIJ": "CGBDJFEI",
    "BCDEFGIK": "CGBDEFIK",
    "BCDEFGIL": "CGBDEFLI",
    "BCDEFGJK": "CGBDJFEK",
    "BCDEFGJL": "CGBDJFLE",
    "BCDEFGKL": "CGBDEFLK",
    "BCDEFHIJ": "CJBDHFEI",
    "BCDEFHIK": "CEBDHFIK",
    "BCDEFHIL": "CEBDHFLI",
    "BCDEFHJK": "CJBDHFEK",
    "BCDEFHJL": "CJBDHFLE",
    "BCDEFHKL": "CEBDHFLK",
    "BCDEFIJK": "CJBDEFIK",
    "BCDEFIJL": "CJBDEFLI",
    "BCDEFIKL": "CEBDIFLK",
    "BCDEFJKL": "CJBDEFLK",
    "BCDEGHIJ": "HGBCJDEI",
    "BCDEGHIK": "EGBCHDIK",
    "BCDEGHIL": "EGBCHDLI",
    "BCDEGHJK": "HGBCJDEK",
    "BCDEGHJL": "HGBCJDLE",
    "BCDEGHKL": "EGBCHDLK",
    "BCDEGIJK": "EGBCJDIK",
    "BCDEGIJL": "EGBCJDLI",
    "BCDEGIKL": "EGBCIDLK",
    "BCDEGJKL": "EGBCJDLK",
    "BCDEHIJK": "EJBCHDIK",
    "BCDEHIJL": "EJBCHDLI",
    "BCDEHIKL": "EIBCHDLK",
    "BCDEHJKL": "EJBCHDLK",
    "BCDEIJKL": "EJBCIDLK",
    "BCDFGHIJ": "HGBCJFDI",
    "BCDFGHIK": "CGBDHFIK",
    "BCDFGHIL": "CGBDHFLI",
    "BCDFGHJK": "HGBCJFDK",
    "BCDFGHJL": "CGBDHFLJ",
    "BCDFGHKL": "CGBDHFLK",
    "BCDFGIJK": "CGBDJFIK",
    "BCDFGIJL": "CGBDJFLI",
    "BCDFGIKL": "CGBDIFLK",
    "BCDFGJKL": "CGBDJFLK",
    "BCDFHIJK": "CJBDHFIK",
    "BCDFHIJL": "CJBDHFLI",
    "BCDFHIKL": "CIBDHFLK",
    "BCDFHJKL": "CJBDHFLK",
    "BCDFIJKL": "CJBDIFLK",
    "BCDGHIJK": "HGBCJDIK",
    "BCDGHIJL": "HGBCJDLI",
    "BCDGHIKL": "HGBCIDLK",
    "BCDGHJKL": "HGBCJDLK",
    "BCDGIJKL": "IGBCJDLK",
    "BCDHIJKL": "HJBCIDLK",
    "BCEFGHIJ": "HGBCJFEI",
    "BCEFGHIK": "EGBCHFIK",
    "BCEFGHIL": "EGBCHFLI",
    "BCEFGHJK": "HGBCJFEK",
    "BCEFGHJL": "HGBCJFLE",
    "BCEFGHKL": "EGBCHFLK",
    "BCEFGIJK": "EGBCJFIK",
    "BCEFGIJL": "EGBCJFLI",
    "BCEFGIKL": "EGBCIFLK",
    "BCEFGJKL": "EGBCJFLK",
    "BCEFHIJK": "EJBCHFIK",
    "BCEFHIJL": "EJBCHFLI",
    "BCEFHIKL": "EIBCHFLK",
    "BCEFHJKL": "EJBCHFLK",
    "BCEFIJKL": "EJBCIFLK",
    "BCEGHIJK": "EJBCHGIK",
    "BCEGHIJL": "EJBCHGLI",
    "BCEGHIKL": "EGBCIHLK",
    "BCEGHJKL": "EJBCHGLK",
    "BCEGIJKL": "EJBCIGLK",
    "BCEHIJKL": "EJBCIHLK",
    "BCFGHIJK": "HGBCJFIK",
    "BCFGHIJL": "HGBCJFLI",
    "BCFGHIKL": "HGBCIFLK",
    "BCFGHJKL": "HGBCJFLK",
    "BCFGIJKL": "IGBCJFLK",
    "BCFHIJKL": "HJBCIFLK",
    "BCGHIJKL": "HJBCIGLK",
    "BDEFGHIJ": "HGBDJFEI",
    "BDEFGHIK": "EGBDHFIK",
    "BDEFGHIL": "EGBDHFLI",
    "BDEFGHJK": "HGBDJFEK",
    "BDEFGHJL": "HGBDJFLE",
    "BDEFGHKL": "EGBDHFLK",
    "BDEFGIJK": "EGBDJFIK",
    "BDEFGIJL": "EGBDJFLI",
    "BDEFGIKL": "EGBDIFLK",
    "BDEFGJKL": "EGBDJFLK",
    "BDEFHIJK": "EJBDHFIK",
    "BDEFHIJL": "EJBDHFLI",
    "BDEFHIKL": "EIBDHFLK",
    "BDEFHJKL": "EJBDHFLK",
    "BDEFIJKL": "EJBDIFLK",
    "BDEGHIJK": "EJBDHGIK",
    "BDEGHIJL": "EJBDHGLI",
    "BDEGHIKL": "EGBDIHLK",
    "BDEGHJKL": "EJBDHGLK",
    "BDEGIJKL": "EJBDIGLK",
    "BDEHIJKL": "EJBDIHLK",
    "BDFGHIJK": "HGBDJFIK",
    "BDFGHIJL": "HGBDJFLI",
    "BDFGHIKL": "HGBDIFLK",
    "BDFGHJKL": "HGBDJFLK",
    "BDFGIJKL": "IGBDJFLK",
    "BDFHIJKL": "HJBDIFLK",
    "BDGHIJKL": "HJBDIGLK",
    "BEFGHIJK": "EJBFHGIK",
    "BEFGHIJL": "EJBFHGLI",
    "BEFGHIKL": "EGBFIHLK",
    "BEFGHJKL": "EJBFHGLK",
    "BEFGIJKL": "EJBFIGLK",
    "BEFHIJKL": "EJBFIHLK",
    "BEGHIJKL": "EJIBHGLK",
    "BFGHIJKL": "HJBFIGLK",
    "CDEFGHIJ": "CGJDHFEI",
    "CDEFGHIK": "CGEDHFIK",
    "CDEFGHIL": "CGEDHFLI",
    "CDEFGHJK": "CGJDHFEK",
    "CDEFGHJL": "CGJDHFLE",
    "CDEFGHKL": "CGEDHFLK",
    "CDEFGIJK": "CGEDJFIK",
    "CDEFGIJL": "CGEDJFLI",
    "CDEFGIKL": "CGEDIFLK",
    "CDEFGJKL": "CGEDJFLK",
    "CDEFHIJK": "CJEDHFIK",
    "CDEFHIJL": "CJEDHFLI",
    "CDEFHIKL": "CEIDHFLK",
    "CDEFHJKL": "CJEDHFLK",
    "CDEFIJKL": "CJEDIFLK",
    "CDEGHIJK": "EGJCHDIK",
    "CDEGHIJL": "EGJCHDLI",
    "CDEGHIKL": "EGICHDLK",
    "CDEGHJKL": "EGJCHDLK",
    "CDEGIJKL": "EGICJDLK",
    "CDEHIJKL": "EJICHDLK",
    "CDFGHIJK": "CGJDHFIK",
    "CDFGHIJL": "CGJDHFLI",
    "CDFGHIKL": "CGIDHFLK",
    "CDFGHJKL": "CGJDHFLK",
    "CDFGIJKL": "CGIDJFLK",
    "CDFHIJKL": "CJIDHFLK",
    "CDGHIJKL": "HGICJDLK",
    "CEFGHIJK": "EGJCHFIK",
    "CEFGHIJL": "EGJCHFLI",
    "CEFGHIKL": "EGICHFLK",
    "CEFGHJKL": "EGJCHFLK",
    "CEFGIJKL": "EGICJFLK",
    "CEFHIJKL": "EJICHFLK",
    "CEGHIJKL": "EJICHGLK",
    "CFGHIJKL": "HGICJFLK",
    "DEFGHIJK": "EGJDHFIK",
    "DEFGHIJL": "EGJDHFLI",
    "DEFGHIKL": "EGIDHFLK",
    "DEFGHJKL": "EGJDHFLK",
    "DEFGIJKL": "EGIDJFLK",
    "DEFHIJKL": "EJIDHFLK",
    "DEGHIJKL": "EJIDHGLK",
    "DFGHIJKL": "HGIDJFLK",
    "EFGHIJKL": "EJIFHGLK",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _alias(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def _load_wc_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (played, fixtures):
    - played: WC 2026 matches already completed (from results_clean.csv)
    - fixtures: remaining unplayed group-stage matches (from fixtures_2026.csv)
    """
    res = pd.read_csv(DATA_DIR / "results_clean.csv")
    res["date"] = pd.to_datetime(res["date"])
    res["home_team"] = res["home_team"].map(_alias).fillna(res["home_team"])
    res["away_team"] = res["away_team"].map(_alias).fillna(res["away_team"])

    wc_played = res[
        (res["tournament"] == "FIFA World Cup") &
        (res["date"].dt.year == 2026) &
        (res["home_score"].notna())
    ].copy()

    fix = pd.read_csv(DATA_DIR / "fixtures_2026.csv")
    fix["date"] = pd.to_datetime(fix["date"])
    fix["home_team"] = fix["home_team"].map(_alias).fillna(fix["home_team"])
    fix["away_team"] = fix["away_team"].map(_alias).fillna(fix["away_team"])

    return wc_played, fix


# ─────────────────────────────────────────────────────────────────────────────
# PMF precomputation
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_pmfs(
    fixtures: pd.DataFrame,
    params: dict,
    max_goals: int = 10,
) -> dict[tuple[str, str], np.ndarray]:
    """Return {(home, away): flat_pmf_vector} for all remaining fixtures."""
    pmfs: dict[tuple[str, str], np.ndarray] = {}
    atk = params["attack"]
    dfn = params["defense"]
    mu  = params["home_advantage"]
    rho = params["rho"]

    def _g(d: dict, k: str) -> float:
        return d.get(k, 0.0)

    for _, row in fixtures.iterrows():
        h, a = row["home_team"], row["away_team"]
        lh = np.exp(mu + _g(atk, h) - _g(dfn, a))
        la = np.exp(_g(atk, a) - _g(dfn, h))
        joint = _dc_joint_pmf(lh, la, rho, max_goals)
        pmfs[(h, a)] = joint.flatten()

    return pmfs


# ─────────────────────────────────────────────────────────────────────────────
# Group stage: standings with FIFA tiebreak rules
# ─────────────────────────────────────────────────────────────────────────────

def _compute_standings(
    group_teams: list[str],
    match_results: list[tuple[str, int, str, int]],
    rng: np.random.Generator,
) -> list[str]:
    """
    Compute group standings and return teams sorted 1st → 4th.

    match_results: list of (home, h_goals, away, a_goals)
    Tiebreaks (FIFA 2026):
      1. Points (3W/1D/0L)
      2. Goal difference (all group games)
      3. Goals scored (all group games)
      4. H2H points (among tied teams only)
      5. H2H goal difference
      6. H2H goals scored
      7. Random (lottery placeholder)
    """
    pts  = defaultdict(int)
    gd   = defaultdict(int)
    gf   = defaultdict(int)

    for h, hg, a, ag in match_results:
        gf[h] += hg; gf[a] += ag
        gd[h] += hg - ag; gd[a] += ag - hg
        if hg > ag:
            pts[h] += 3
        elif ag > hg:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1

    def h2h_sort_key(tied_teams):
        """For a set of tied teams, return h2h (pts, gd, gf) per team."""
        h2h_pts  = defaultdict(int)
        h2h_gd_  = defaultdict(int)
        h2h_gf_  = defaultdict(int)
        tied_set = set(tied_teams)
        for h, hg, a, ag in match_results:
            if h in tied_set and a in tied_set:
                h2h_gf_[h] += hg; h2h_gf_[a] += ag
                h2h_gd_[h] += hg - ag; h2h_gd_[a] += ag - hg
                if hg > ag:   h2h_pts[h] += 3
                elif ag > hg: h2h_pts[a] += 3
                else:         h2h_pts[h] += 1; h2h_pts[a] += 1
        return h2h_pts, h2h_gd_, h2h_gf_

    def sort_key(team, h2h_pts, h2h_gd_, h2h_gf_):
        rand = rng.integers(0, 10**9)
        return (
            -pts[team],
            -gd[team],
            -gf[team],
            -h2h_pts[team],
            -h2h_gd_[team],
            -h2h_gf_[team],
            int(rand),
        )

    # Sort with group-level criteria first, then H2H for ties
    by_pts = sorted(group_teams, key=lambda t: (-pts[t], -gd[t], -gf[t]))

    # Resolve ties with H2H within each tier
    result = []
    i = 0
    while i < len(by_pts):
        j = i + 1
        while j < len(by_pts) and (
            pts[by_pts[j]] == pts[by_pts[i]] and
            gd[by_pts[j]]  == gd[by_pts[i]]  and
            gf[by_pts[j]]  == gf[by_pts[i]]
        ):
            j += 1
        tied = by_pts[i:j]
        if len(tied) > 1:
            h2h_pts, h2h_gd_, h2h_gf_ = h2h_sort_key(tied)
            tied.sort(key=lambda t: sort_key(t, h2h_pts, h2h_gd_, h2h_gf_))
        result.extend(tied)
        i = j

    return result


def _third_place_rank_key(stats: dict, team: str) -> tuple:
    """Tiebreak key for ranking 3rd-place teams across groups."""
    return (-stats[team]["pts"], -stats[team]["gd"], -stats[team]["gf"])


def _compute_team_stats(
    group_teams: list[str],
    match_results: list[tuple[str, int, str, int]],
) -> dict[str, dict]:
    """Return {team: {pts, gd, gf}} for all teams in a group."""
    pts: dict[str, int] = {t: 0 for t in group_teams}
    gd:  dict[str, int] = {t: 0 for t in group_teams}
    gf:  dict[str, int] = {t: 0 for t in group_teams}
    for h, hg, a, ag in match_results:
        if h in pts:
            gf[h] += hg; gd[h] += hg - ag
            if hg > ag:   pts[h] += 3
            elif hg == ag: pts[h] += 1
        if a in pts:
            gf[a] += ag; gd[a] += ag - hg
            if ag > hg:   pts[a] += 3
            elif hg == ag: pts[a] += 1
    return {t: {"pts": pts[t], "gd": gd[t], "gf": gf[t]} for t in group_teams}


# ─────────────────────────────────────────────────────────────────────────────
# 495-combination lookup
# ─────────────────────────────────────────────────────────────────────────────

def _get_third_seedings(qualifying_groups: list[str]) -> dict[int, str]:
    """
    Given 8 group letters whose 3rd-place teams advanced,
    return {match_id: group_letter_of_3rd_place_opponent}.
    """
    key = "".join(sorted(qualifying_groups))
    assignment = _COMBO_DICT.get(key)
    if assignment is None:
        raise ValueError(f"No combo entry for qualifying groups {sorted(qualifying_groups)}")

    seedings: dict[int, str] = {}
    for col_idx, match_id in enumerate(_THIRD_COL_TO_MATCH):
        seedings[match_id] = assignment[col_idx]

    return seedings


# ─────────────────────────────────────────────────────────────────────────────
# Slot resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_slot(
    slot: str,
    group_standings: dict[str, list[str]],
    third_place_teams: dict[str, str],  # group_letter → team
    third_seedings:    dict[int, str],  # match_id → group_letter
    match_id: int,
) -> str:
    """
    slot examples: "1A", "2B", or the match_id (int) for a 3rd-place slot.
    third_seedings is passed for context, but for 3rd-place slots we use
    the team from third_place_teams[group_letter].
    """
    if slot.startswith(("1", "2")):
        pos   = int(slot[0]) - 1     # 0=1st, 1=2nd
        group = slot[1]
        return group_standings[group][pos]
    raise ValueError(f"Unexpected slot: {slot}")


# ─────────────────────────────────────────────────────────────────────────────
# Knockout match simulation
# ─────────────────────────────────────────────────────────────────────────────

def _sim_ko_match(
    team1: str,
    team2: str,
    params: dict,
    rng: np.random.Generator,
    max_goals: int = 10,
) -> str:
    """
    Simulate a knockout match (all neutral). Return the winner.
    If tied after 90 min: simulate 30 min extra time, then weighted
    coin-flip for penalties.
    """
    atk = params["attack"]
    dfn = params["defense"]
    rho = params["rho"]

    def _g(d, k):
        return d.get(k, 0.0)

    # 90-minute rates
    lh = np.exp(_g(atk, team1) - _g(dfn, team2))
    la = np.exp(_g(atk, team2) - _g(dfn, team1))

    joint = _dc_joint_pmf(lh, la, rho, max_goals)
    flat  = joint.flatten()
    idx   = rng.choice(len(flat), p=flat)
    hg    = idx // (max_goals + 1)
    ag    = idx %  (max_goals + 1)

    if hg != ag:
        return team1 if hg > ag else team2

    # Extra time: Poisson with 1/3 of 90-min rate (30 min)
    et_lh = lh / 3.0
    et_la = la / 3.0
    et_hg = rng.poisson(et_lh)
    et_ag = rng.poisson(et_la)
    if et_hg != et_ag:
        return team1 if et_hg > et_ag else team2

    # Penalties: weighted by relative expected goals
    denom = lh + la
    p_home = lh / denom if denom > 0 else 0.5
    return team1 if rng.random() < p_home else team2


# ─────────────────────────────────────────────────────────────────────────────
# Single simulation run
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_once(
    played:    pd.DataFrame,
    fixtures:  pd.DataFrame,
    pmfs:      dict[tuple[str, str], np.ndarray],
    params:    dict,
    rng:       np.random.Generator,
    max_goals: int = 10,
) -> tuple[dict[str, str], dict[str, dict]]:
    """
    Run one full tournament simulation.

    Returns
    -------
    reached : {team: furthest_round_reached}
        Round ∈ {"Group","R32","R16","QF","SF","Final","Champion"}
    group_info : {team: {"pos": int, "pts": int, "gd": int, "gf": int,
                          "group_qualified": bool}}
        Finishing position (1–4) and cumulative group-stage stats per team.
        "group_qualified" is True if the team advances to R32 (1st/2nd or
        best-8 3rd-place).
    """
    # ── Group stage: collect results ─────────────────────────────────────────
    group_results: dict[str, list[tuple]] = {g: [] for g in GROUPS}

    # Actual played matches
    for _, row in played.iterrows():
        h = row["home_team"]; a = row["away_team"]
        if h not in TEAM_GROUP or a not in TEAM_GROUP:
            continue
        hg = int(row["home_score"]); ag = int(row["away_score"])
        g  = TEAM_GROUP[h]
        group_results[g].append((h, hg, a, ag))

    # Simulate remaining fixtures
    for _, row in fixtures.iterrows():
        h = row["home_team"]; a = row["away_team"]
        if (h, a) not in pmfs:
            continue
        flat = pmfs[(h, a)]
        idx  = rng.choice(len(flat), p=flat)
        hg   = idx // (max_goals + 1)
        ag   = idx %  (max_goals + 1)
        g    = TEAM_GROUP.get(h)
        if g:
            group_results[g].append((h, hg, a, ag))

    # ── Compute standings ────────────────────────────────────────────────────
    group_standings: dict[str, list[str]] = {}
    third_stats: dict[str, dict] = {}  # group_letter → {pts, gd, gf}

    for g, teams in GROUPS.items():
        sorted_teams = _compute_standings(teams, group_results[g], rng)
        group_standings[g] = sorted_teams

        # Third-place team stats (for cross-group ranking)
        third = sorted_teams[2]
        pts_total = sum(
            (3 if h == third and hg > ag else
             3 if a == third and ag > hg else
             1 if hg == ag and (h == third or a == third) else 0)
            for h, hg, a, ag in group_results[g]
        )
        gd_total = sum(
            (hg - ag if h == third else ag - hg)
            for h, hg, a, ag in group_results[g]
            if h == third or a == third
        )
        gf_total = sum(
            (hg if h == third else ag)
            for h, hg, a, ag in group_results[g]
            if h == third or a == third
        )
        third_stats[g] = {"pts": pts_total, "gd": gd_total, "gf": gf_total}

    # ── 3rd-place ranking: top 8 of 12 advance ───────────────────────────────
    sorted_thirds = sorted(
        GROUPS.keys(),
        key=lambda g: _third_place_rank_key(third_stats, g),
    )
    qualifying_third_groups = sorted_thirds[:8]
    qualifying_third_set    = set(qualifying_third_groups)

    # Map group letter → 3rd-place team name
    third_place_teams: dict[str, str] = {
        g: group_standings[g][2] for g in qualifying_third_groups
    }

    # ── Per-team group stage snapshot ────────────────────────────────────────
    group_info: dict[str, dict] = {}
    for g, teams in GROUPS.items():
        team_stats = _compute_team_stats(teams, group_results[g])
        for pos, team in enumerate(group_standings[g], start=1):
            group_info[team] = {
                "pos":             pos,
                "pts":             team_stats[team]["pts"],
                "gd":              team_stats[team]["gd"],
                "gf":              team_stats[team]["gf"],
                "group_qualified": (pos <= 2) or (pos == 3 and g in qualifying_third_set),
            }

    # 495-combo lookup: which 3rd-place team faces which group winner
    third_seedings = _get_third_seedings(qualifying_third_groups)

    # ── R32 seeding ──────────────────────────────────────────────────────────
    # match_id → (team1, team2)
    ko_matchups: dict[int, tuple[str, str]] = {}

    for match_id, slot1, slot2 in R32_FIXED:
        t1 = _resolve_slot(slot1, group_standings, third_place_teams, third_seedings, match_id)
        t2 = _resolve_slot(slot2, group_standings, third_place_teams, third_seedings, match_id)
        ko_matchups[match_id] = (t1, t2)

    for match_id, confirmed_slot in R32_THIRD.items():
        t1 = _resolve_slot(confirmed_slot, group_standings, third_place_teams, third_seedings, match_id)
        third_group = third_seedings[match_id]
        t2 = third_place_teams[third_group]
        ko_matchups[match_id] = (t1, t2)

    # ── Knockout rounds ───────────────────────────────────────────────────────
    # Track advancement: team → furthest round reached
    reached: dict[str, str] = {}
    for g, teams in GROUPS.items():
        for t in teams:
            reached[t] = "Group"

    # All 32 R32 qualifiers advance to at least R32
    for t1, t2 in ko_matchups.values():
        reached[t1] = "R32"
        reached[t2] = "R32"

    # winner[match_id] = winning team; loser[match_id] = losing team
    winner: dict[int, str] = {}
    loser:  dict[int, str] = {}

    def _advance(team: str, rnd: str) -> None:
        if team:
            reached[team] = rnd

    # ── Play R32 (all 16 ko_matchups matches, sorted by match_id) ─────────
    for match_id in sorted(ko_matchups.keys()):
        t1, t2 = ko_matchups[match_id]
        w = _sim_ko_match(t1, t2, params, rng)
        l = t2 if w == t1 else t1
        winner[match_id] = w
        loser[match_id]  = l
        _advance(w, "R16")

    # ── Play R16 ──────────────────────────────────────────────────────────
    for match_id, feed1, feed2 in R16:
        t1 = winner[feed1]; t2 = winner[feed2]
        w = _sim_ko_match(t1, t2, params, rng)
        l = t2 if w == t1 else t1
        winner[match_id] = w
        loser[match_id]  = l
        _advance(w, "QF")

    # ── Play QF ───────────────────────────────────────────────────────────
    for match_id, feed1, feed2 in QF:
        t1 = winner[feed1]; t2 = winner[feed2]
        w = _sim_ko_match(t1, t2, params, rng)
        l = t2 if w == t1 else t1
        winner[match_id] = w
        loser[match_id]  = l
        _advance(w, "SF")

    # ── Play SF ───────────────────────────────────────────────────────────
    for match_id, feed1, feed2 in SF:
        t1 = winner[feed1]; t2 = winner[feed2]
        w = _sim_ko_match(t1, t2, params, rng)
        l = t2 if w == t1 else t1
        winner[match_id] = w
        loser[match_id]  = l
        _advance(w, "Final")

    # ── 3rd-place match ───────────────────────────────────────────────────
    tp_id, sf1_id, sf2_id = THIRD_PLACE_MATCH
    t1 = loser[sf1_id]; t2 = loser[sf2_id]
    w3 = _sim_ko_match(t1, t2, params, rng)
    winner[tp_id] = w3

    # ── Final ─────────────────────────────────────────────────────────────
    fin_id, sf1_id, sf2_id = FINAL
    t1 = winner[sf1_id]; t2 = winner[sf2_id]
    champ = _sim_ko_match(t1, t2, params, rng)
    _advance(champ, "Champion")

    return reached, group_info


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run(
    n_sims:      int = 10_000,
    seed:        int | None = None,
    params_path: pathlib.Path | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """
    Run the full tournament simulation n_sims times.

    Returns
    -------
    tournament_probs : dict[team, {round: probability}]
        Probability of reaching each knockout round.
        Keys: "R32", "R16", "QF", "SF", "Final", "Champion"

    group_probs : dict[team, {stat: value}]
        Group-stage statistics aggregated over all simulations:
        - "P1st", "P2nd", "P3rd", "P4th"  — probability of each finish
        - "P_qual"  — probability of advancing to R32 (top-2 or best-8 3rd)
        - "avg_pts" — expected points (3/1/0 per match)
        - "avg_gd"  — expected goal difference
        - "avg_gf"  — expected goals scored
    """
    if params_path is None:
        params_path = DATA_DIR / "dc_params_2016.json"

    params  = load_params(params_path)
    played, fixtures = _load_wc_matches()
    pmfs    = _precompute_pmfs(fixtures, params)
    rng     = np.random.default_rng(seed)

    ROUND_ORDER = ["Group", "R32", "R16", "QF", "SF", "Final", "Champion"]
    rank        = {r: i for i, r in enumerate(ROUND_ORDER)}
    advance_counts: dict[str, dict[str, int]]   = defaultdict(lambda: defaultdict(int))
    group_pos_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    group_stat_sums:  dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    group_qual_count: dict[str, int]             = defaultdict(int)

    print(f"Running {n_sims:,} simulations …", end="", flush=True)
    for i in range(n_sims):
        if i % 1000 == 0 and i > 0:
            print(f" {i//1000}k", end="", flush=True)
        reached, group_info = _simulate_once(played, fixtures, pmfs, params, rng)

        # Accumulate knockout round advancement
        for team, rnd in reached.items():
            r_idx = rank[rnd]
            for r in ROUND_ORDER[1:]:
                if rank[r] <= r_idx:
                    advance_counts[team][r] += 1

        # Accumulate group stage data
        for team, gdata in group_info.items():
            group_pos_counts[team][gdata["pos"]] += 1
            group_stat_sums[team]["pts"] += gdata["pts"]
            group_stat_sums[team]["gd"]  += gdata["gd"]
            group_stat_sums[team]["gf"]  += gdata["gf"]
            if gdata["group_qualified"]:
                group_qual_count[team] += 1

    print(" done.")

    all_teams = sorted({t for ts in GROUPS.values() for t in ts})

    tournament_probs: dict[str, dict[str, float]] = {}
    for team in all_teams:
        tournament_probs[team] = {
            r: advance_counts[team].get(r, 0) / n_sims
            for r in ROUND_NAMES
        }

    group_probs: dict[str, dict[str, float]] = {}
    for team in all_teams:
        group_probs[team] = {
            "P1st":    group_pos_counts[team].get(1, 0) / n_sims,
            "P2nd":    group_pos_counts[team].get(2, 0) / n_sims,
            "P3rd":    group_pos_counts[team].get(3, 0) / n_sims,
            "P4th":    group_pos_counts[team].get(4, 0) / n_sims,
            "P_qual":  group_qual_count[team] / n_sims,
            "avg_pts": group_stat_sums[team]["pts"] / n_sims,
            "avg_gd":  group_stat_sums[team]["gd"]  / n_sims,
            "avg_gf":  group_stat_sums[team]["gf"]  / n_sims,
        }

    return tournament_probs, group_probs


def run_group_stage(
    n_sims:      int = 10_000,
    seed:        int | None = None,
    params_path: pathlib.Path | None = None,
) -> dict[str, dict[str, float]]:
    """
    Convenience wrapper: run simulations and return only group-stage stats.
    See `run()` for full return value documentation.
    """
    _, group_probs = run(n_sims=n_sims, seed=seed, params_path=params_path)
    return group_probs


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print output
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: dict[str, dict[str, float]], top_n: int = 48) -> None:
    header = f"{'Team':<28} {'R32':>6} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>7} {'Champion':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    ranked = sorted(results.items(), key=lambda x: -x[1]["Champion"])
    for team, probs in ranked[:top_n]:
        grp = TEAM_GROUP.get(team, "?")
        row = (
            f"{team:<28}"
            f" {probs['R32']:>5.1%}"
            f" {probs['R16']:>5.1%}"
            f" {probs['QF']:>5.1%}"
            f" {probs['SF']:>5.1%}"
            f" {probs['Final']:>6.1%}"
            f" {probs['Champion']:>8.1%}"
        )
        print(row)
    print("=" * len(header))


def print_group_stage_results(group_probs: dict[str, dict[str, float]]) -> None:
    """
    Print group-stage performance tables, one block per group (A–L).

    Columns
    -------
    1st / 2nd / 3rd / 4th  — probability of finishing in that position
    Qualify                 — probability of advancing to R32
    AvgPts                  — expected points across 3 group matches
    AvgGD                   — expected goal difference
    AvgGF                   — expected goals scored
    """
    col_hdr = (
        f"  {'Team':<28}"
        f" {'1st':>5} {'2nd':>5} {'3rd':>5} {'4th':>5}"
        f" {'Qualify':>8}"
        f" {'AvgPts':>7} {'AvgGD':>7} {'AvgGF':>7}"
    )
    sep_inner = "  " + "─" * (len(col_hdr) - 2)

    for g_letter in sorted(GROUPS.keys()):
        teams = GROUPS[g_letter]
        print(f"\n{'─'*60}")
        print(f"  Group {g_letter}")
        print(col_hdr)
        print(sep_inner)

        sorted_teams = sorted(teams, key=lambda t: -group_probs[t]["P_qual"])
        for team in sorted_teams:
            s = group_probs[team]
            print(
                f"  {team:<28}"
                f" {s['P1st']:>4.1%}"
                f" {s['P2nd']:>4.1%}"
                f" {s['P3rd']:>4.1%}"
                f" {s['P4th']:>4.1%}"
                f" {s['P_qual']:>7.1%}"
                f" {s['avg_pts']:>7.2f}"
                f" {s['avg_gd']:>+7.2f}"
                f" {s['avg_gf']:>7.2f}"
            )
    print(f"\n{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FIFA 2026 World Cup Monte Carlo simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/simulate.py                            # full tournament, 10k sims
  python src/simulate.py --group-stage              # show group-stage tables too
  python src/simulate.py --group-stage --only       # group-stage tables only
  python src/simulate.py --sims 50000 --seed 7      # reproducible high-precision
  python src/simulate.py --out data/sim_results.json
  python src/simulate.py --out-groups data/group_results.json
        """,
    )
    parser.add_argument("--sims",        type=int, default=10_000,
                        help="Number of simulations (default: 10,000)")
    parser.add_argument("--seed",        type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--params",      type=str,
                        default=str(DATA_DIR / "dc_params_2016.json"),
                        help="Path to dc_params JSON")
    parser.add_argument("--out",         type=str, default=None,
                        help="Save tournament results as JSON to this path")
    parser.add_argument("--out-groups",  type=str, default=None,
                        help="Save group-stage results as JSON to this path")
    parser.add_argument("--group-stage", action="store_true",
                        help="Print group-stage position/qualification tables")
    parser.add_argument("--only",        action="store_true",
                        help="With --group-stage: skip the full knockout table")
    args = parser.parse_args()

    tournament_probs, group_probs = run(
        n_sims      = args.sims,
        seed        = args.seed,
        params_path = pathlib.Path(args.params),
    )

    if args.group_stage:
        print_group_stage_results(group_probs)

    if not (args.group_stage and args.only):
        print_results(tournament_probs)

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(tournament_probs, f, indent=2)
        print(f"\nTournament results saved to {out_path}")

    if args.out_groups:
        out_path = pathlib.Path(args.out_groups)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(group_probs, f, indent=2)
        print(f"Group-stage results saved to {out_path}")


if __name__ == "__main__":
    main()
