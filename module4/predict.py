#!/usr/bin/env python3
"""One-off command-line use, no server:
       python predict.py my_lunch.jpg
Downloads the model on first run, then works offline."""
import sys, json
from PIL import Image
import app as A

if len(sys.argv) < 2:
    sys.exit("usage: python predict.py <image.jpg>")
A.load_artifacts()
res = A.analyze(Image.open(sys.argv[1]))
t = res["totals"]
print("\n" + "="*58)
print(f"TOTAL: {t['calories']:.0f} kcal | fat {t['fat_g']:.1f} g | "
      f"carbs {t['carbs_g']:.1f} g | protein {t['protein_g']:.1f} g")
print("="*58)
print(f"{'INGREDIENT':22s}{'STATUS':10s}{'GRAMS':>7s}{'KCAL':>7s}")
for r in res["ingredients"]:
    print(f"{r['name'][:21]:22s}{r['status']:10s}{r['grams']:7.1f}{r['kcal']:7.1f}")
print("="*58)
print("check 9f+4c+4p =", res["atwater_check"]["from_macros"], "== kcal", res["atwater_check"]["calories"])
