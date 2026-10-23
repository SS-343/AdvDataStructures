"""
impact.py - From "what happens next" to "what will it cost": a discrete-event
Monte Carlo of the rest of the intrusion on a real network graph.

For each simulated incident, starting at the moment of our third clue:
  1. Draw the attacker type from the POSTERIOR given the clues (exact, per outer draw).
  2. Run a discrete-event simulation driven by a PRIORITY QUEUE (heapq min-heap):
       - "step" events: the attacker's next tactic, at a lognormal dwell time later
       - one "contain" event: when the security team isolates the intrusion
     The earliest event always pops first. If containment pops before the attacker
     reaches an objective, the attack is stopped.
  3. Lateral Movement steps move the attacker across the network graph (network.py):
     with probability P_TARGETED it takes the BFS shortest-path hop toward its
     crown jewel, otherwise it moves to a random neighbouring host.
  4. Draw a financial loss from a lognormal distribution whose MEAN comes from the
     IBM Cost of a Data Breach Report 2025 where a figure exists, scaled up if the
     attacker reached its crown jewel.

Repeating this tens of thousands of times gives a loss distribution and the
Loss Exceedance Curve P(loss > $x).
"""

import heapq

import numpy as np

from model import ACTORS, DWELL_MEDIAN_H, DWELL_SIGMA, IDX, N_STATES, N_TACTICS, evidence_likelihood
from network import FOOTHOLD, TARGET, frontier, next_hop_toward
from simulate import sample_matrices

# ----------------------------------------------------------------------------- losses (USD)
# Mean loss per objective. Sources: IBM Cost of a Data Breach Report 2025 (global figures).
LOSS_MEAN = {
    "EXF": 4.44e6,    # global average cost of a data breach (IBM 2025)
    "RAN": 5.08e6,    # ransomware / extortion disclosed by the attacker (IBM 2025)
    "DES": 5.08e6,    # ASSUMPTION: no IBM figure; set equal to ransomware
    "CRED": 1.11e6,   # ASSUMPTION: credentials stolen, no data loss; 25% of the breach average
}
CONTAINED_MEAN = 0.25e6   # ASSUMPTION: investigation + clean-up when stopped before an objective
LOSS_SIGMA = 1.0          # lognormal spread (heavy right tail)
JEWEL_MULT = {False: 0.7, True: 1.3}  # crown jewel reached -> worse incident
P_TARGETED = 0.6          # chance a lateral move follows the BFS shortest path to the target

# Time until the security team contains the intrusion, from now (hours, lognormal).
CONTAIN = {
    "Slow response (median 72 h)": 72.0,
    "Current response (median 24 h)": 24.0,
    "Fast response (median 6 h)": 6.0,
}
CONTAIN_SIGMA = 0.8
OBJ_ORDER = ["EXF", "RAN", "DES", "CRED"]


def lognormal_with_mean(rng, mean, sigma):
    return rng.lognormal(np.log(mean) - sigma ** 2 / 2, sigma)


def simulate_incident(P, actor, rng, contain_median_h):
    """One discrete-event incident from the third clue onward. Returns a dict."""
    cum = np.cumsum(P[actor], axis=1)
    cum[:, -1] = 1.0
    target = TARGET[ACTORS[actor]]
    state = IDX["LM"]
    owned = set(FOOTHOLD)                       # hash set of compromised hosts
    log_dwell = np.log(DWELL_MEDIAN_H[actor])

    events = []                                 # min-heap of (time_h, seq, kind)
    heapq.heappush(events, (rng.lognormal(np.log(contain_median_h), CONTAIN_SIGMA), 0, "contain"))
    heapq.heappush(events, (rng.lognormal(log_dwell, DWELL_SIGMA), 1, "step"))
    seq = 2
    while True:
        t, _, kind = heapq.heappop(events)
        if kind == "contain":
            return {"objective": None, "t": t, "hosts": len(owned), "jewel": target in owned}
        nxt = int(np.searchsorted(cum[state], rng.random(), side="left"))
        nxt = min(nxt, N_STATES - 1)
        if nxt == IDX["LM"]:
            hop = next_hop_toward(owned, target) if rng.random() < P_TARGETED else None
            if hop is None:
                f = frontier(owned)
                hop = f[rng.integers(len(f))] if f else None
            if hop is not None:
                owned.add(hop)
        if nxt >= N_TACTICS:
            return {"objective": nxt - N_TACTICS, "t": t, "hosts": len(owned), "jewel": target in owned}
        state = nxt
        heapq.heappush(events, (t + rng.lognormal(log_dwell, DWELL_SIGMA), seq, "step"))
        seq += 1


def run_impact(evidence, prior, contain_median_h, kappa=50, outer=200, inner=200, seed=0):
    """Nested MC: outer = parameter draws (posterior computed exactly per draw), inner = incidents."""
    rng = np.random.default_rng(seed)
    out = {"loss": [], "objective": [], "jewel": [], "hosts": [], "t": []}
    per_draw = {"exp_loss": [], "p_obj": []}
    for _ in range(outer):
        P = sample_matrices(rng, kappa)
        like = np.array([evidence_likelihood(P[a], evidence) for a in range(len(ACTORS))])
        post = prior * like
        post /= post.sum()
        losses, hit = [], 0
        for _ in range(inner):
            a = rng.choice(len(ACTORS), p=post)
            r = simulate_incident(P, a, rng, contain_median_h)
            if r["objective"] is None:
                loss = lognormal_with_mean(rng, CONTAINED_MEAN, LOSS_SIGMA)
                out["objective"].append(-1)
            else:
                mean = LOSS_MEAN[OBJ_ORDER[r["objective"]]] * JEWEL_MULT[r["jewel"]]
                loss = lognormal_with_mean(rng, mean, LOSS_SIGMA)
                out["objective"].append(r["objective"])
                hit += 1
            losses.append(loss)
            out["jewel"].append(r["jewel"]); out["hosts"].append(r["hosts"]); out["t"].append(r["t"])
        out["loss"].extend(losses)
        per_draw["exp_loss"].append(np.mean(losses))
        per_draw["p_obj"].append(hit / inner)
    return {k: np.array(v) for k, v in out.items()}, {k: np.array(v) for k, v in per_draw.items()}
