"""
simulate.py - The Monte Carlo engine.

Two nested layers of randomness:

  OUTER loop (epistemic uncertainty - "we don't know the true probabilities"):
      For each outer draw d = 1..D, sample a plausible set of transition matrices,
      one per actor, from a Dirichlet centred on the model's estimates:
          row ~ Dirichlet(kappa * p_row)
      kappa is our confidence: large kappa -> rows stay close to the estimate.

  INNER loop (aleatory uncertainty - "attackers make random choices"):
      Simulate N complete intrusions:
        1. draw the actor type from the prior
        2. start at Initial Access and random-walk the chain until an objective
           is reached, adding a random dwell time for every step.
      Each step is detected by the SOC with probability q[tactic]. Keep (accept)
      only intrusions whose DETECTED steps so far are exactly what the SOC has
      observed (undetected steps may happen in between). This is rejection
      sampling, i.e. Monte Carlo Bayesian conditioning on the evidence.

For every accepted intrusion we record: the actor, the tactic taken immediately
after the last observed event, the final objective, and the hours remaining
until the objective.
"""

import numpy as np
from model import ACTORS, BASE_P, DWELL_MEDIAN_H, DWELL_SIGMA, IDX, N_STATES, N_TACTICS, detect_vector


def sample_matrices(rng, kappa, base=BASE_P):
    """One Dirichlet draw of every actor's transition matrix (zero entries stay zero)."""
    if kappa is None or np.isinf(kappa):
        return base.copy()
    P = base.copy()
    for a in range(base.shape[0]):
        for s in range(N_TACTICS):
            support = base[a, s] > 0
            alpha = kappa * base[a, s, support]
            P[a, s, support] = rng.dirichlet(alpha)
    return P


def simulate_paths(P, prior, evidence, n, rng, max_steps=200):
    """Simulate n intrusions under fixed matrices P; return dict for accepted paths."""
    ev = np.array([IDX[e] for e in evidence])
    L = len(ev)
    assert ev[0] == IDX["IA"], "evidence must start at Initial Access"

    cum = np.cumsum(P, axis=2)
    cum[..., -1] = 1.0

    actor = rng.choice(len(prior), size=n, p=prior)
    state = np.full(n, IDX["IA"])
    ptr = np.ones(n, dtype=int)              # evidence items matched so far
    matched = ptr == L
    next_tac = np.full(n, -1)
    steps_left = np.zeros(n, dtype=int)
    hours_left = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    log_med = np.log(DWELL_MEDIAN_H)
    q = np.append(detect_vector(), np.zeros(N_STATES - N_TACTICS))
    rejected = np.zeros(n, dtype=bool)

    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        u = rng.random(idx.size)
        rows = cum[actor[idx], state[idx]]                    # (k, S)
        nxt = (rows < u[:, None]).sum(axis=1)                 # inverse-CDF sampling

        m = matched[idx]
        # paths already past the evidence: record next step + accumulate time to objective
        first = m & (next_tac[idx] == -1)
        next_tac[idx[first]] = nxt[first]
        mi = idx[m]
        steps_left[mi] += 1
        hours_left[mi] += rng.lognormal(log_med[actor[mi]], DWELL_SIGMA)

        # paths still matching the evidence: is this step detected?
        um = ~m
        seen = um & (rng.random(idx.size) < q[nxt])
        want = ev[np.minimum(ptr[idx], L - 1)]
        hit = seen & (nxt == want)
        ptr[idx[hit]] += 1
        matched[idx[hit & (ptr[idx] == L)]] = True
        bad = seen & (nxt != want)                            # detected something we did NOT see
        rejected[idx[bad]] = True

        state[idx] = nxt
        alive[idx] = (nxt < N_TACTICS) & ~rejected[idx]

    ok = matched & ~alive & ~rejected                         # consistent with evidence & finished
    return {
        "n_sim": n,
        "actor": actor[ok],
        "next": next_tac[ok],
        "objective": state[ok] - N_TACTICS,
        "steps": steps_left[ok],
        "hours": hours_left[ok],
    }


def run_nested(evidence, prior, kappa=50, outer=300, inner=20000, seed=0):
    """Nested Monte Carlo. Returns per-outer-draw estimates plus pooled samples."""
    rng = np.random.default_rng(seed)
    A = len(ACTORS)
    per_draw = {"post": [], "obj": [], "next": [], "p24": [], "accept": []}
    pooled = {k: [] for k in ("actor", "next", "objective", "steps", "hours")}
    for _ in range(outer):
        P = sample_matrices(rng, kappa)
        r = simulate_paths(P, prior, evidence, inner, rng)
        k = len(r["actor"])
        if k == 0:
            continue
        per_draw["accept"].append(k / inner)
        per_draw["post"].append(np.bincount(r["actor"], minlength=A) / k)
        per_draw["obj"].append(np.bincount(r["objective"], minlength=4) / k)
        per_draw["next"].append(np.bincount(r["next"], minlength=N_STATES) / k)
        per_draw["p24"].append(np.mean(r["hours"] <= 24))
        for key in pooled:
            pooled[key].append(r[key])
    per_draw = {k: np.array(v) for k, v in per_draw.items()}
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    return per_draw, pooled
