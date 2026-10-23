"""
experiments.py - Runs every experiment in the project and writes results + charts.

    python experiments.py            # ~1-2 minutes on a laptop

Outputs:  results/results.json  and  results/fig_*.png

Experiments
  E1  Verification   - MC with fixed matrices vs the exact Markov-chain solution,
                       plus convergence of the estimate as the number of runs grows.
  E2  Main result    - the slide scenario (phishing -> stolen password -> new server),
                       nested MC with parameter uncertainty: who, next step, objective, time.
  E3  Evidence       - how the prediction changes as more steps are detected.
  E4  Sensitivity    - to the actor prior (industry) and to confidence in the matrices (kappa).
  E5  Impact         - discrete-event simulation on a network graph -> financial loss and
                       Loss Exceedance Curves for three security-team response speeds.
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from model import (ACTOR_SHORT, ACTORS, BASE_P, CODES, IDX, N_TACTICS, NAMES, OBJ_CODES,
                   PRIORS, exact_prediction)
from simulate import run_nested, simulate_paths

OUT = "results"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- style
matplotlib.rcParams.update({
    "font.family": "Inter", "font.size": 15, "axes.titlesize": 17,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#0b0b0b",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
# Objective colours (fixed order); each actor wears the colour of its typical objective.
OBJ_COL = {"EXF": "#2a78d6", "RAN": "#eb6834", "DES": "#1baf7a", "CRED": "#eda100"}
ACTOR_COL = ["#eb6834", "#eda100", "#2a78d6", "#1baf7a"]
OBJ_LABEL = {"EXF": "Data exfiltration", "RAN": "Ransomware",
             "DES": "Destruction", "CRED": "Credential theft"}

MAIN_EVIDENCE = ["IA", "CA", "LM"]
BASE_PRIOR = PRIORS["Baseline enterprise"]


def ci(a, lo=5, hi=95):
    return np.percentile(a, lo, axis=0), np.percentile(a, hi, axis=0)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


results = {}

# ============================================================================ E1
print("E1 verification ...")
post_x, obj_x, next_x = exact_prediction(BASE_P, BASE_PRIOR, MAIN_EVIDENCE)
rng = np.random.default_rng(7)
r = simulate_paths(BASE_P, BASE_PRIOR, MAIN_EVIDENCE, 1_000_000, rng)
k = len(r["actor"])
obj_mc = np.bincount(r["objective"], minlength=4) / k
post_mc = np.bincount(r["actor"], minlength=4) / k
se = np.sqrt(obj_mc * (1 - obj_mc) / k)
results["verification"] = {
    "simulated": 1_000_000, "accepted": int(k), "acceptance_rate": k / 1_000_000,
    "objective_exact": dict(zip(OBJ_CODES, obj_x.round(4).tolist())),
    "objective_mc": dict(zip(OBJ_CODES, obj_mc.round(4).tolist())),
    "objective_mc_se": dict(zip(OBJ_CODES, se.round(4).tolist())),
    "posterior_exact": dict(zip(ACTORS, post_x.round(4).tolist())),
    "posterior_mc": dict(zip(ACTORS, post_mc.round(4).tolist())),
    "max_abs_error": float(np.max(np.abs(obj_mc - obj_x))),
}

# convergence: running estimate of P(credential theft) as accepted runs accumulate
target = OBJ_CODES.index("CRED")
hits = (r["objective"] == target).astype(float)
ns = np.unique(np.logspace(1, np.log10(k), 60).astype(int))
est = np.cumsum(hits)[ns - 1] / ns
band = 1.96 * np.sqrt(est * (1 - est) / ns)
results["convergence"] = {"n": ns.tolist(), "estimate": est.round(4).tolist(),
                          "exact": float(obj_x[target])}

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.fill_between(ns, est - band, est + band, color="#2a78d6", alpha=0.18, linewidth=0,
                label="95% confidence band")
ax.plot(ns, est, color="#2a78d6", lw=2, label="Monte Carlo estimate")
ax.axhline(obj_x[target], color=INK, lw=1.2, ls=(0, (4, 3)), label="Exact answer (linear algebra)")
ax.set_xscale("log")
ax.set_ylim(0, 0.8)
ax.set_xlabel("Accepted simulated intrusions (log scale)")
ax.set_ylabel("P(credential theft)")
ax.grid(axis="y")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.legend(frameon=False, loc="upper right")
save(fig, "fig_convergence.png")

# ============================================================================ E2
print("E2 main nested run ...")
per, pool = run_nested(MAIN_EVIDENCE, BASE_PRIOR, kappa=50, outer=400, inner=40000, seed=11)
obj_mean = per["obj"].mean(0); obj_lo, obj_hi = ci(per["obj"])
post_mean = per["post"].mean(0); post_lo, post_hi = ci(per["post"])
nxt_mean = per["next"].mean(0); nxt_lo, nxt_hi = ci(per["next"])
hours = pool["hours"]; objs = pool["objective"]

time_by_obj = {}
for j, c in enumerate(OBJ_CODES):
    h = hours[objs == j]
    time_by_obj[c] = {"median_h": float(np.median(h)), "p25_h": float(np.percentile(h, 25)),
                      "p75_h": float(np.percentile(h, 75)), "p_within_24h": float(np.mean(h <= 24)),
                      "median_steps": float(np.median(pool["steps"][objs == j]))}
p24_lo, p24_hi = ci(per["p24"])
results["main"] = {
    "evidence": [NAMES[e] for e in MAIN_EVIDENCE],
    "outer_draws": int(len(per["obj"])), "inner_paths": 40000, "kappa": 50,
    "total_simulated": int(len(per["obj"]) * 40000), "total_accepted": int(len(objs)),
    "mean_acceptance": float(per["accept"].mean()),
    "objective": {c: {"mean": float(obj_mean[j]), "lo90": float(obj_lo[j]), "hi90": float(obj_hi[j])}
                  for j, c in enumerate(OBJ_CODES)},
    "posterior": {a: {"prior": float(BASE_PRIOR[i]), "mean": float(post_mean[i]),
                      "lo90": float(post_lo[i]), "hi90": float(post_hi[i])}
                  for i, a in enumerate(ACTORS)},
    "next_step": {CODES[s]: {"mean": float(nxt_mean[s]), "lo90": float(nxt_lo[s]), "hi90": float(nxt_hi[s])}
                  for s in range(len(CODES)) if nxt_mean[s] > 0.001},
    "p_objective_within_24h": {"mean": float(per["p24"].mean()), "lo90": float(p24_lo), "hi90": float(p24_hi)},
    "hours_to_objective_median": float(np.median(hours)),
    "time_by_objective": time_by_obj,
}

# --- chart: final objective with 90% interval
order = np.argsort(obj_mean)
fig, ax = plt.subplots(figsize=(11, 4.6))
y = np.arange(4)
for yi, j in zip(y, order):
    c = OBJ_CODES[j]
    ax.barh(yi, obj_mean[j], height=0.56, color=OBJ_COL[c])
    ax.plot([obj_lo[j], obj_hi[j]], [yi, yi], color=INK, lw=1.6)
    for xv in (obj_lo[j], obj_hi[j]):
        ax.plot([xv, xv], [yi - 0.12, yi + 0.12], color=INK, lw=1.6)
    ax.text(obj_hi[j] + 0.012, yi, f"{obj_mean[j]:.0%}", va="center", color=INK, fontsize=17,
            fontweight="semibold")
    ax.text(obj_hi[j] + 0.075, yi, f"(90% interval {obj_lo[j]:.0%}–{obj_hi[j]:.0%})", va="center",
            color=INK2, fontsize=13)
ax.set_yticks(y, [OBJ_LABEL[OBJ_CODES[j]] for j in order])
ax.set_xlim(0, 0.75)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.grid(axis="x"); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Probability this is the attacker's final objective")
save(fig, "fig_objective.png")

# --- chart: prior vs posterior actor
fig, ax = plt.subplots(figsize=(11, 4.6))
x = np.arange(4); w = 0.36
ax.bar(x - w / 2, BASE_PRIOR, w, color="#c3c2b7")
for i in range(4):
    ax.bar(x[i] + w / 2, post_mean[i], w, color=ACTOR_COL[i])
    ax.plot([x[i] + w / 2] * 2, [post_lo[i], post_hi[i]], color=INK, lw=1.6)
    ax.text(x[i] + w / 2, post_hi[i] + 0.015, f"{post_mean[i]:.0%}", ha="center", color=INK,
            fontsize=15, fontweight="semibold")
    ax.text(x[i] - w / 2, BASE_PRIOR[i] + 0.015, f"{BASE_PRIOR[i]:.0%}", ha="center", color=INK2,
            fontsize=14)
ax.set_xticks(x, ACTOR_SHORT)
ax.set_ylim(0, 0.55)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.grid(axis="y"); ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
ax.text(0.99, 0.97, "Gray bar = before evidence (prior)    Colored bar = after evidence (posterior, 90% interval)",
        transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=14)
save(fig, "fig_posterior.png")

# --- chart: next step
nz = [s for s in range(len(CODES)) if nxt_mean[s] > 0.001]
nz.sort(key=lambda s: nxt_mean[s])
fig, ax = plt.subplots(figsize=(11, 5.6))
for yi, s in enumerate(nz):
    is_obj = s >= N_TACTICS
    col = OBJ_COL[CODES[s]] if is_obj else "#898781"
    ax.barh(yi, nxt_mean[s], height=0.6, color=col)
    ax.plot([nxt_lo[s], nxt_hi[s]], [yi, yi], color=INK, lw=1.4)
    ax.text(nxt_hi[s] + 0.008, yi, f"{nxt_mean[s]:.0%}", va="center", color=INK, fontsize=14,
            fontweight="semibold")
labels = [(OBJ_LABEL[CODES[s]] + " (objective)" if s >= N_TACTICS else NAMES[CODES[s]]) for s in nz]
ax.set_yticks(range(len(nz)), labels)
ax.set_xlim(0, 0.36)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.grid(axis="x"); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Probability of the attacker's very next step")
save(fig, "fig_next.png")

# --- chart: time to objective (ECDF per objective)
fig, ax = plt.subplots(figsize=(11, 5.0))
for j, c in enumerate(OBJ_CODES):
    h = np.sort(hours[objs == j])
    ax.plot(h, np.arange(1, len(h) + 1) / len(h), color=OBJ_COL[c], lw=2.2, label=OBJ_LABEL[c])
ax.axvline(24, color=INK, lw=1.2, ls=(0, (4, 3)))
ax.text(26, 0.04, "24 hours", color=INK, fontsize=14)
ax.set_xscale("log"); ax.set_xlim(1, 2000)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}h"))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Hours from now until the objective is reached (log scale)")
ax.set_ylabel("Share of simulated attacks")
ax.grid(True); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_time.png")

# ============================================================================ E3
print("E3 evidence progression ...")
SCEN = [
    ("Phishing email opened", ["IA"]),
    ("+ stolen password used", ["IA", "CA"]),
    ("+ login to a new server", ["IA", "CA", "LM"]),
    ("+ security tools disabled", ["IA", "CA", "LM", "DE"]),
    ("+ files staged / archived", ["IA", "CA", "LM", "CO"]),
]
prog = []
for label, ev in SCEN:
    p, _ = run_nested(ev, BASE_PRIOR, kappa=50, outer=200, inner=30000, seed=21)
    m = p["obj"].mean(0); lo, hi = ci(p["obj"])
    post = p["post"].mean(0)
    prog.append({"label": label, "evidence": ev,
                 "objective": {c: {"mean": float(m[j]), "lo90": float(lo[j]), "hi90": float(hi[j])}
                               for j, c in enumerate(OBJ_CODES)},
                 "posterior": dict(zip(ACTORS, post.round(4).tolist())),
                 "acceptance": float(p["accept"].mean())})
results["evidence_progression"] = prog


def stacked(rows, labels, fname, highlight=None):
    fig, ax = plt.subplots(figsize=(12, 0.95 * len(rows) + 1.6))
    for yi, row in enumerate(rows):
        left = 0
        for c in OBJ_CODES:
            v = row[c]
            ax.barh(yi, v, left=left, height=0.62, color=OBJ_COL[c], edgecolor="#fcfcfb", linewidth=2,
                    label=OBJ_LABEL[c] if yi == 0 else None)
            if v >= 0.07:
                ax.text(left + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=14,
                        color=INK if c == "CRED" else "#ffffff", fontweight="semibold")
            left += v
    ax.set_yticks(range(len(rows)), labels)
    if highlight is not None:
        ax.get_yticklabels()[highlight].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="y", length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=14,
              handlelength=1.0)
    save(fig, fname)


stacked([{c: s["objective"][c]["mean"] for c in OBJ_CODES} for s in prog],
        [s["label"] for s in prog], "fig_progression.png", highlight=2)

# ============================================================================ E4
print("E4 sensitivity ...")
sens_prior = []
for name, pr in PRIORS.items():
    p, _ = run_nested(MAIN_EVIDENCE, pr, kappa=50, outer=200, inner=30000, seed=31)
    m = p["obj"].mean(0)
    sens_prior.append({"prior": name, "weights": dict(zip(ACTORS, pr.tolist())),
                       "objective": dict(zip(OBJ_CODES, m.round(4).tolist()))})
results["sensitivity_prior"] = sens_prior
stacked([s["objective"] for s in sens_prior], [s["prior"] for s in sens_prior],
        "fig_sensitivity_prior.png", highlight=0)

sens_kappa = []
for kap in [5, 20, 50, 200, np.inf]:
    p, _ = run_nested(MAIN_EVIDENCE, BASE_PRIOR, kappa=kap, outer=200, inner=30000, seed=41)
    m = p["obj"].mean(0); lo, hi = ci(p["obj"])
    sens_kappa.append({"kappa": "fixed (no parameter uncertainty)" if np.isinf(kap) else int(kap),
                       "objective": {c: {"mean": float(m[j]), "lo90": float(lo[j]), "hi90": float(hi[j]),
                                         "width": float(hi[j] - lo[j])} for j, c in enumerate(OBJ_CODES)}})
results["sensitivity_kappa"] = sens_kappa

# ============================================================================ E5
print("E5 network + financial loss ...")
from impact import CONTAIN, JEWEL_MULT, LOSS_MEAN, CONTAINED_MEAN, P_TARGETED, run_impact
from network import ADJ, FOOTHOLD, HOSTS, TARGET, _EDGES, bfs_hops

hops = bfs_hops(FOOTHOLD)
loss_res = {"assumptions": {"loss_mean_usd": LOSS_MEAN, "contained_mean_usd": CONTAINED_MEAN,
                            "jewel_multiplier": {str(k): v for k, v in JEWEL_MULT.items()},
                            "p_targeted_move": P_TARGETED,
                            "containment_median_h": CONTAIN},
            "network": {"hosts": len(HOSTS), "links": len(_EDGES),
                        "bfs_hops_from_foothold": {h: hops[h] for h in ("DC1", "DB1", "BKP1")}},
            "scenarios": []}
curves = {}
for name, med in CONTAIN.items():
    o, p = run_impact(MAIN_EVIDENCE, BASE_PRIOR, med, kappa=50, outer=200, inner=300, seed=51)
    L = o["loss"]
    lo, hi = ci(p["p_obj"])
    sc = {"name": name, "contain_median_h": med, "incidents": int(len(L)),
          "p_objective_before_containment": {"mean": float(p["p_obj"].mean()), "lo90": float(lo), "hi90": float(hi)},
          "expected_loss": float(L.mean()), "median_loss": float(np.median(L)),
          "p95_loss": float(np.percentile(L, 95)), "p99_loss": float(np.percentile(L, 99)),
          "p_loss_over_1m": float(np.mean(L > 1e6)), "p_loss_over_5m": float(np.mean(L > 5e6)),
          "p_loss_over_10m": float(np.mean(L > 1e7)),
          "crown_jewel_reached": float(o["jewel"].mean()), "mean_hosts_owned": float(o["hosts"].mean())}
    loss_res["scenarios"].append(sc)
    curves[name] = np.sort(L)
base = loss_res["scenarios"][1]["expected_loss"]
for sc in loss_res["scenarios"]:
    sc["expected_loss_change_vs_current"] = sc["expected_loss"] - base
results["loss"] = loss_res

# --- chart: loss exceedance curves
fig, ax = plt.subplots(figsize=(11, 5.2))
cols = ["#86b6ef", "#2a78d6", "#104281"]
for (name, L), c in zip(curves.items(), cols):
    xs = np.logspace(4, 8, 400)
    ys = 1 - np.searchsorted(L, xs, side="right") / len(L)
    ax.plot(xs, ys, color=c, lw=2.4, label=name)
for xv in (1e6, 5e6):
    ax.axvline(xv, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.set_xscale("log"); ax.set_xlim(1e4, 1e8); ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"${v/1e6:g}M" if v >= 1e6 else f"${v/1e3:g}k"))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Loss from this incident (log scale)")
ax.set_ylabel("Chance the loss is larger")
ax.grid(True); ax.set_axisbelow(True)
ax.set_xticks([1e4, 1e5, 1e6, 5e6, 1e7, 1e8])
ax.legend(loc="upper right", frameon=True, facecolor="#fcfcfb", edgecolor="#fcfcfb", framealpha=1)
save(fig, "fig_lec.png")

# --- chart: the network
POS = {}
for i in range(1, 11):
    ang = np.deg2rad(90 - (i - 1) * 36)
    POS[f"WS{i:02d}"] = (2.0 + 1.7 * np.cos(ang), 3.0 + 1.7 * np.sin(ang))
POS.update({"MAIL1": (2.0, 3.0), "PRT1": (-0.5, 1.2), "FS1": (5.3, 1.9), "APP1": (5.3, 3.9),
            "APP2": (7.0, 4.8), "WEB1": (7.0, 3.4), "JUMP1": (7.0, 2.0), "HR1": (7.4, 0.5),
            "VPN1": (8.6, 3.5), "DB2": (10.0, 4.8), "DB1": (11.4, 3.8), "DC1": (10.0, 2.0),
            "DC2": (11.4, 1.0), "BKP1": (12.6, 2.4)})
fig, ax = plt.subplots(figsize=(12, 5.6))
for (x0, x1, lab) in [(-1.0, 4.2, "User network"), (4.5, 9.2, "Server network"), (9.4, 13.3, "Core: identity, data, backups")]:
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x0, -0.35), x1 - x0, 5.95, boxstyle="round,pad=0,rounding_size=0.25",
                 facecolor="#f2f3f5", edgecolor="#e1e0d9", lw=1, zorder=0))
    ax.text((x0 + x1) / 2, 5.4, lab, ha="center", va="center", color=INK2, fontsize=14)
for a, b in _EDGES:
    (xa, ya), (xb, yb) = POS[a], POS[b]
    ax.plot([xa, xb], [ya, yb], color="#c3c2b7", lw=1.1, zorder=1)
for h, (x, y) in POS.items():
    role = HOSTS[h][0]
    if h in FOOTHOLD:
        face, edge, size = "#2a78d6", "#fcfcfb", 420
    elif role == "crown":
        face, edge, size = "#fcfcfb", "#eb6834", 520
    else:
        face, edge, size = ("#898781" if role == "server" else "#c3c2b7"), "#fcfcfb", (300 if role == "server" else 180)
    ax.scatter([x], [y], s=size, color=face, edgecolors=edge, linewidths=3 if role == "crown" else 2, zorder=3)
    dy = -0.42 if h not in ("MAIL1",) else -0.38
    ax.text(x, y + dy, h, ha="center", va="top", fontsize=11.5, color=INK, zorder=4)
for h, lab, above in [("DC1", "identity", False), ("DB1", "data", True), ("BKP1", "backups", False)]:
    x, y = POS[h]
    ax.text(x, y + (0.42 if above else -0.82), f"{lab} · {hops[h]} hops", ha="center",
            va="bottom" if above else "top", fontsize=13, color="#b8461b", fontweight="semibold", zorder=4,
            bbox=dict(facecolor="#f2f3f5", edgecolor="none", pad=1.5))
ax.scatter([], [], s=180, color="#2a78d6", label="Already compromised (our clues)")
ax.scatter([], [], s=180, color="#fcfcfb", edgecolors="#eb6834", linewidths=3, label="Crown jewel (BFS hops away)")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=13)
ax.set_xlim(-1.2, 13.5); ax.set_ylim(-0.6, 5.8); ax.set_aspect("equal"); ax.axis("off")
save(fig, "fig_network.png")

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("done ->", OUT)
