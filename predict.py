"""
predict.py - Quick command-line prediction for any observed sequence of tactics.

    python predict.py IA CA LM
    python predict.py IA CA LM CO --prior "Defense contractor (APT-heavy)" --kappa 20

Tactic codes: IA EX PE PR DE CA DI LM CO C2
"""

import argparse

import numpy as np

from model import ACTORS, CODES, IDX, NAMES, OBJ_CODES, PRIORS, TACTICS
from simulate import run_nested

ap = argparse.ArgumentParser()
ap.add_argument("evidence", nargs="+", help="detected tactics in order, starting with IA")
ap.add_argument("--prior", default="Baseline enterprise", choices=list(PRIORS))
ap.add_argument("--kappa", type=float, default=50)
ap.add_argument("--outer", type=int, default=200)
ap.add_argument("--inner", type=int, default=20000)
args = ap.parse_args()

valid = {c for c, _ in TACTICS}
for e in args.evidence:
    if e not in valid:
        ap.error(f"unknown tactic code {e!r}; use one of {sorted(valid)}")

per, pool = run_nested(args.evidence, PRIORS[args.prior], args.kappa, args.outer, args.inner, seed=1)
lo = lambda a: np.percentile(a, 5, axis=0)
hi = lambda a: np.percentile(a, 95, axis=0)

print(f"\nObserved: {' -> '.join(NAMES[e] for e in args.evidence)}   (prior: {args.prior})")
print(f"Simulated {len(per['obj']) * args.inner:,} intrusions, {len(pool['objective']):,} matched the evidence\n")

print("Who is it?")
for i, a in enumerate(ACTORS):
    print(f"  {a:28s} {per['post'].mean(0)[i]:6.1%}   [{lo(per['post'])[i]:.0%}-{hi(per['post'])[i]:.0%}]")

print("\nFinal objective")
m = per["obj"].mean(0)
for j in np.argsort(-m):
    print(f"  {NAMES[OBJ_CODES[j]]:28s} {m[j]:6.1%}   [{lo(per['obj'])[j]:.0%}-{hi(per['obj'])[j]:.0%}]")

print("\nMost likely next step")
n = per["next"].mean(0)
for s in np.argsort(-n)[:5]:
    print(f"  {NAMES[CODES[s]]:28s} {n[s]:6.1%}")

print(f"\nP(objective reached within 24h): {per['p24'].mean():.0%}")
print(f"Median hours until objective:    {np.median(pool['hours']):.0f}h\n")
