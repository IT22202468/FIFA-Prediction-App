"""
FIFA World Cup 2026 — Full Tournament Simulator
================================================
Simulates the complete Canada/USA/Mexico 2026 World Cup N times and reports
per-team probabilities for every round.

Format
------
  48 teams  •  12 groups of 4  •  Top 2 + 8 best 3rd per group → Round of 32
  R32 (matches 73-88) → R16 (89-96) → QF (97-100) → SF (101-102)
  3rd-place (103)  •  Final (104)

Groups (2026 WC confirmed draw — June 2026)
--------------------------------------------
  A: Mexico, South Korea, Czech Republic, South Africa
  B: Canada, Switzerland, Bosnia and Herzegovina, Qatar
  C: Brazil, Morocco, Scotland, Haiti
  D: United States, Australia, Paraguay, Turkey
  E: Germany, Côte d'Ivoire, Ecuador, Curaçao
  F: Netherlands, Japan, Sweden, Tunisia
  G: Belgium, Egypt, Iran, New Zealand
  H: Spain, Cabo Verde, Saudi Arabia, Uruguay
  I: France, Senegal, Iraq, Norway
  J: Argentina, Algeria, Austria, Jordan
  K: Portugal, DR Congo, Uzbekistan, Colombia
  L: England, Croatia, Ghana, Panama

R32 bracket (matches 73-88)
----------------------------
  Fixed slots:
    M73: 2A vs 2B       M75: 1F vs 2C       M76: 1C vs 2F
    M78: 2E vs 2I       M83: 2K vs 2L       M84: 1H vs 2J
    M86: 1J vs 2H       M88: 2D vs 2G
  Third-place slots (opponent determined by 495-combo table):
    M74: 1E vs 3rd(ABCDF)   M77: 1I vs 3rd(CDFGH)   M79: 1A vs 3rd(CEFHI)
    M80: 1L vs 3rd(EHIJK)   M81: 1D vs 3rd(BEFIJ)   M82: 1G vs 3rd(AEHIJ)
    M85: 1B vs 3rd(EFGIJ)   M87: 1K vs 3rd(DEIJL)

R16: M89(W74/W77) M90(W73/W75) M91(W76/W78) M92(W79/W80)
     M93(W83/W84) M94(W81/W82) M95(W86/W88) M96(W85/W87)
QF:  M97(W89/W90) M98(W93/W94) M99(W91/W92) M100(W95/W96)
SF:  M101(W97/W98)  M102(W99/W100)
3rd: M103(L101/L102)   Final: M104(W101/W102)

Usage
-----
  python src/simulate_2026.py
  python src/simulate_2026.py --sims 50000 --seed 42
  python src/simulate_2026.py --out data/sim_2026.json
  python src/simulate_2026.py --list-teams
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))
from predict import load_params, _dc_joint_pmf

MAX_GOALS = 10

# ---------------------------------------------------------------------------
# Tournament structure
# ---------------------------------------------------------------------------

GROUPS = {
    "A": ["Czech Republic", "Mexico", "South Africa", "South Korea"],
    "B": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"],
    "C": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "D": ["Australia", "Paraguay", "Turkey", "United States"],
    "E": ["Côte d'Ivoire", "Curaçao", "Ecuador", "Germany"],
    "F": ["Japan", "Netherlands", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Cabo Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "I": ["France", "Iraq", "Norway", "Senegal"],
    "J": ["Algeria", "Argentina", "Austria", "Jordan"],
    "K": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "L": ["Croatia", "England", "Ghana", "Panama"],
}

# Aliases for alternative spellings that may appear in the raw data
TEAM_ALIASES = {
    "Czechia":         "Czech Republic",
    "Türkiye":         "Turkey",
    "Turkiye":         "Turkey",
    "Ivory Coast":     "Côte d'Ivoire",
    "Cape Verde":      "Cabo Verde",
    "Curacao":         "Curaçao",
    "IR Iran":         "Iran",
    "Korea Republic":  "South Korea",
    "Congo DR":        "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}

# ---------------------------------------------------------------------------
# 3rd-place seeding — 495-combo lookup table
# ---------------------------------------------------------------------------
# Column order maps to R32 match slots: [79, 85, 81, 74, 82, 77, 87, 80]
# i.e. value[0]→M79, value[1]→M85, value[2]→M81, value[3]→M74,
#      value[4]→M82, value[5]→M77, value[6]→M87, value[7]→M80
_THIRD_COL_TO_MATCH = [79, 85, 81, 74, 82, 77, 87, 80]

# Pre-built lookup: sorted(8 qualifying group letters) → 8-char assignment string
_COMBO_DICT = {
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
# ---------------------------------------------------------------------------
# R32 bracket definition
# ---------------------------------------------------------------------------

# Fixed matchups (group slot vs group slot)
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

# Third-place matchups: {match_id: group_winner_slot}
# The third-place opponent is determined at runtime via _COMBO_DICT
R32_THIRD_FIXED = {
    74: "1E",
    77: "1I",
    79: "1A",
    80: "1L",
    81: "1D",
    82: "1G",
    85: "1B",
    87: "1K",
}

# R16: (match_id, r32_match_1, r32_match_2)
R16 = [
    (89, 74, 77),
    (90, 73, 75),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]

# QF/SF/Final
QF    = [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)]
SF    = [(101, 97, 98), (102, 99, 100)]
THIRD = (103, 101, 102)
FINAL = (104, 101, 102)

ALL_ROUNDS = ["R32", "R16", "QF", "SF", "Final", "Champion"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_aliases(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def _all_teams():
    return [_apply_aliases(t) for teams in GROUPS.values() for t in teams]


def _group_fixtures():
    """Return list of (home, away, group) for every group match."""
    fixtures = []
    for g, teams in GROUPS.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                fixtures.append((_apply_aliases(teams[i]),
                                 _apply_aliases(teams[j]),
                                 g))
    return fixtures


# ---------------------------------------------------------------------------
# PMF precomputation
# ---------------------------------------------------------------------------

def _precompute_pmfs(fixtures, params):
    atk = params["attack"]
    dfn = params["defense"]
    rho = params["rho"]

    def sa(t): return atk.get(t, 0.0)
    def sd(t): return dfn.get(t, 0.0)

    pmfs = {}
    unknown = set()
    for h, a, _ in fixtures:
        for t in (h, a):
            if t not in atk:
                unknown.add(t)
        key = (h, a)
        if key not in pmfs:
            lh = np.exp(sa(h) - sd(a))
            la = np.exp(sa(a) - sd(h))
            pmfs[key] = _dc_joint_pmf(lh, la, rho, MAX_GOALS).flatten()
    if unknown:
        print(f"  ⚠  Teams not in model (using default 0): {sorted(unknown)}")
    return pmfs


# ---------------------------------------------------------------------------
# Group-stage standings (FIFA tiebreak)
# ---------------------------------------------------------------------------

def _standings(teams, results, rng):
    """Sort teams 1st→4th using FIFA tiebreak rules."""
    pts = defaultdict(int); gd = defaultdict(int); gf = defaultdict(int)
    for h, hg, a, ag in results:
        gf[h] += hg; gf[a] += ag
        gd[h] += hg - ag; gd[a] += ag - hg
        if   hg > ag: pts[h] += 3
        elif ag > hg: pts[a] += 3
        else:         pts[h] += 1; pts[a] += 1

    def h2h(tied):
        s = set(tied)
        hp = defaultdict(int); hd = defaultdict(int); hf = defaultdict(int)
        for h, hg, a, ag in results:
            if h in s and a in s:
                hf[h] += hg; hf[a] += ag
                hd[h] += hg - ag; hd[a] += ag - hg
                if   hg > ag: hp[h] += 3
                elif ag > hg: hp[a] += 3
                else:         hp[h] += 1; hp[a] += 1
        return hp, hd, hf

    order = sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t]))
    out = []
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and \
              pts[order[j]] == pts[order[i]] and \
              gd[order[j]]  == gd[order[i]]  and \
              gf[order[j]]  == gf[order[i]]:
            j += 1
        tied = order[i:j]
        if len(tied) > 1:
            hp, hd, hf = h2h(tied)
            tied.sort(key=lambda t: (-hp[t], -hd[t], -hf[t],
                                     int(rng.integers(0, 10**9))))
        out.extend(tied)
        i = j
    return out, pts[out[0]], gd[out[0]], gf[out[0]]


# ---------------------------------------------------------------------------
# Knockout match (extra time + penalties)
# ---------------------------------------------------------------------------

def _sim_ko(t1, t2, params, rng):
    atk = params["attack"]
    dfn = params["defense"]
    rho = params["rho"]

    def sa(t): return atk.get(t, 0.0)
    def sd(t): return dfn.get(t, 0.0)

    # 90 min (neutral)
    lh = np.exp(sa(t1) - sd(t2))
    la = np.exp(sa(t2) - sd(t1))
    flat = _dc_joint_pmf(lh, la, rho, MAX_GOALS).flatten()
    idx  = rng.choice(len(flat), p=flat)
    g1, g2 = divmod(idx, MAX_GOALS + 1)
    if g1 != g2:
        return (t1 if g1 > g2 else t2)

    # Extra time (30 min ≈ ⅓ of 90 min)
    flat_et = _dc_joint_pmf(lh / 3, la / 3, rho, MAX_GOALS).flatten()
    idx     = rng.choice(len(flat_et), p=flat_et)
    g1, g2  = divmod(idx, MAX_GOALS + 1)
    if g1 != g2:
        return (t1 if g1 > g2 else t2)

    # Penalties — weighted coin flip by attack rating
    s1 = np.exp(sa(t1)); s2 = np.exp(sa(t2))
    return t1 if rng.random() < s1 / (s1 + s2) else t2


# ---------------------------------------------------------------------------
# Single full-tournament simulation
# ---------------------------------------------------------------------------

def _simulate_once(fixtures, pmfs, params, rng):
    N1 = MAX_GOALS + 1

    # ── Group stage ──────────────────────────────────────────────────────
    bucket = defaultdict(list)   # group → [(h, hg, a, ag)]
    for h, a, g in fixtures:
        flat = pmfs[(h, a)]
        idx  = rng.choice(len(flat), p=flat)
        hg, ag = divmod(idx, N1)
        bucket[g].append((h, hg, a, ag))

    # Resolve standings; track 3rd-place team stats for best-8 selection
    group_slot = {}   # "1A"/"2A"/"3A" → team name
    third_stats = {}  # group_letter → (pts, gd, gf, team)

    for g, teams in GROUPS.items():
        team_list = [_apply_aliases(t) for t in teams]
        sorted_teams, pts3, gd3, gf3 = _standings(team_list, bucket[g], rng)
        group_slot[f"1{g}"] = sorted_teams[0]
        group_slot[f"2{g}"] = sorted_teams[1]
        group_slot[f"3{g}"] = sorted_teams[2]
        third_stats[g] = (pts3, gd3, gf3, sorted_teams[2])

    # ── Pick 8 best 3rd-place teams ────────────────────────────────────
    # Sort all 12 third-place teams by pts → gd → gf → random draw
    all_thirds = sorted(
        third_stats.items(),
        key=lambda kv: (-kv[1][0], -kv[1][1], -kv[1][2],
                        int(rng.integers(0, 10**9)))
    )
    best8_groups = [g for g, _ in all_thirds[:8]]  # e.g. ["A","B","C","D","E","F","G","H"]
    best8_teams  = {g: third_stats[g][3] for g in best8_groups}

    # Look up 495-combo seeding assignment
    combo_key = ''.join(sorted(best8_groups))
    assignment = _COMBO_DICT.get(combo_key)
    if assignment is None:
        # Fallback: assign in sorted order to the 8 slots
        assignment = ''.join(sorted(best8_groups))

    # Map each R32 third-place slot to the actual team
    # _THIRD_COL_TO_MATCH = [79, 85, 81, 74, 82, 77, 87, 80]
    third_slot_team = {}  # match_id → 3rd-place team
    for col_idx, match_id in enumerate(_THIRD_COL_TO_MATCH):
        assigned_group = assignment[col_idx]
        third_slot_team[match_id] = best8_teams.get(assigned_group, "Unknown")

    # ── Knockout ─────────────────────────────────────────────────────────
    winner = {}   # match_id → winning team
    loser  = {}   # match_id → losing team
    reached = defaultdict(str)  # team → deepest round reached

    def _play(match_id, t1, t2):
        w = _sim_ko(t1, t2, params, rng)
        l = t2 if w == t1 else t1
        winner[match_id] = w
        loser[match_id]  = l
        return w, l

    # R32 — fixed slots
    for mid, s1, s2 in R32_FIXED:
        t1 = group_slot[s1]
        t2 = group_slot[s2]
        w, l = _play(mid, t1, t2)
        reached[w] = "R32"; reached[l] = "R32"

    # R32 — third-place slots
    for mid, winner_slot in R32_THIRD_FIXED.items():
        t1 = group_slot[winner_slot]
        t2 = third_slot_team[mid]
        w, l = _play(mid, t1, t2)
        reached[w] = "R32"; reached[l] = "R32"

    # R16
    for mid, f1, f2 in R16:
        w, l = _play(mid, winner[f1], winner[f2])
        reached[w] = "R16"; reached[l] = "R16"

    # QF
    for mid, f1, f2 in QF:
        w, l = _play(mid, winner[f1], winner[f2])
        reached[w] = "QF"; reached[l] = "QF"

    # SF
    sf_losers = []
    for mid, f1, f2 in SF:
        w, l = _play(mid, winner[f1], winner[f2])
        reached[w] = "SF"; reached[l] = "SF"
        sf_losers.append(l)

    # 3rd-place
    _play(THIRD[0], sf_losers[0], sf_losers[1])

    # Final
    w, l = _play(FINAL[0], winner[SF[0][0]], winner[SF[1][0]])
    reached[w] = "Champion"; reached[l] = "Final"

    return reached


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(n_sims=10_000, seed=None, params_path=None):
    if params_path is None:
        params_path = DATA_DIR / "dc_params.json"

    params   = load_params(params_path)
    fixtures = _group_fixtures()
    pmfs     = _precompute_pmfs(fixtures, params)
    rng      = np.random.default_rng(seed)

    all_teams = _all_teams()
    counts    = defaultdict(lambda: defaultdict(int))  # team → round → count

    print(f"FIFA World Cup 2026 — {n_sims:,} simulations")
    if len(_COMBO_DICT) < 495:
        print(f"  ⚠  Only {len(_COMBO_DICT)}/495 combo entries loaded — "
              f"3rd-place seeding may fall back to sorted assignment")
    print(f"Running...", end="", flush=True)

    for i in range(n_sims):
        if (i + 1) % 10_000 == 0:
            print(f" {(i+1)//1_000}k", end="", flush=True)
        reached = _simulate_once(fixtures, pmfs, params, rng)
        for team, deepest in reached.items():
            depth = ALL_ROUNDS.index(deepest)
            for r in ALL_ROUNDS[:depth + 1]:
                counts[team][r] += 1

    print(" done.")

    results = {}
    for team in sorted(all_teams):
        results[team] = {r: round(counts[team][r] / n_sims, 4) for r in ALL_ROUNDS}

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(results, top_n=48):
    sorted_teams = sorted(results, key=lambda t: -results[t].get("Champion", 0))[:top_n]

    print(f"\n{'─' * 90}")
    print(f"  {'Team':<28}  {'R32':>6}  {'R16':>6}  {'QF':>6}  "
          f"{'SF':>6}  {'Final':>6}  {'Champion':>8}")
    print(f"  {'─'*28}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}")

    for team in sorted_teams:
        d = results[team]
        print(f"  {team:<28}  "
              f"{d['R32']*100:5.1f}%  "
              f"{d['R16']*100:5.1f}%  "
              f"{d['QF']*100:5.1f}%  "
              f"{d['SF']*100:5.1f}%  "
              f"{d['Final']*100:5.1f}%  "
              f"{d['Champion']*100:7.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Simulate the 2026 FIFA World Cup")
    ap.add_argument("--sims",       type=int, default=10_000)
    ap.add_argument("--seed",       type=int, default=None)
    ap.add_argument("--params",     type=str, default=None)
    ap.add_argument("--out",        type=str, default=None)
    ap.add_argument("--list-teams", action="store_true")
    ap.add_argument("--combo-count", action="store_true",
                    help="Print the number of loaded combo-table entries and exit")
    args = ap.parse_args()

    params_path = pathlib.Path(args.params) if args.params else DATA_DIR / "dc_params.json"

    if args.combo_count:
        print(f"Combo dict entries loaded: {len(_COMBO_DICT)} / 495")
        return

    if args.list_teams:
        params = load_params(params_path)
        for t in sorted(params["attack"]):
            print(f"  {t}")
        return

    results = run(n_sims=args.sims, seed=args.seed, params_path=params_path)
    print_results(results)

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved to {p}")


if __name__ == "__main__":
    main()
