"""
model.py - The attacker model: states, threat-actor profiles, and transition matrices.

The attacker is modelled as an absorbing Markov chain over MITRE ATT&CK tactics.
  * 10 transient states  = ATT&CK tactics the attacker moves between (and can loop through)
  * 4 absorbing states   = the attacker's final objective (what we want to predict)

Each threat-actor type has its own transition matrix. All actors share the same
"base" tradecraft (how intrusions generally unfold); each actor then adds an overlay
that biases it toward its own objective (e.g. ransomware crews move laterally fast
and encrypt; espionage crews collect and exfiltrate slowly).

IMPORTANT: The weights below are ILLUSTRATIVE, expert-style estimates chosen to
reflect the qualitative patterns described in public threat reporting. They are not
fitted to a dataset. The simulation treats them as uncertain (see simulate.py,
Dirichlet sampling), and the sensitivity analysis shows how much the answers depend
on them. Replacing them with frequencies counted from real incident data (e.g. MITRE
ATT&CK group/campaign technique mappings, CISA advisories, or an organisation's own
incident tickets) is the natural next step.
"""

import numpy as np

# --------------------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------------------
TACTICS = [
    ("IA", "Initial Access"),
    ("EX", "Execution"),
    ("PE", "Persistence"),
    ("PR", "Privilege Escalation"),
    ("DE", "Defense Evasion"),
    ("CA", "Credential Access"),
    ("DI", "Discovery"),
    ("LM", "Lateral Movement"),
    ("CO", "Collection"),
    ("C2", "Command & Control"),
]
OBJECTIVES = [
    ("EXF", "Data Exfiltration"),
    ("RAN", "Ransomware"),
    ("DES", "Destruction (wiper)"),
    ("CRED", "Credential Theft"),
]
STATES = TACTICS + OBJECTIVES
CODES = [c for c, _ in STATES]
NAMES = {c: n for c, n in STATES}
IDX = {c: i for i, c in enumerate(CODES)}
N_TACTICS = len(TACTICS)
N_STATES = len(STATES)
OBJ_CODES = [c for c, _ in OBJECTIVES]

# --------------------------------------------------------------------------------------
# Threat-actor types
# --------------------------------------------------------------------------------------
ACTORS = ["Ransomware affiliate", "Access / credential broker", "Espionage APT", "Destructive (wiper)"]
ACTOR_SHORT = ["Ransomware", "Cred. broker", "Espionage APT", "Wiper"]

# Prior probability of each actor type for a generic enterprise (assumption).
PRIORS = {
    "Baseline enterprise": np.array([0.35, 0.30, 0.20, 0.15]),
    "Hospital (ransomware-heavy)": np.array([0.60, 0.25, 0.05, 0.10]),
    "Defense contractor (APT-heavy)": np.array([0.20, 0.20, 0.50, 0.10]),
    "Uninformed (uniform)": np.array([0.25, 0.25, 0.25, 0.25]),
}

# Probability the SOC detects (logs/alerts on) a step in each tactic. Illustrative.
# Observed evidence = the sequence of DETECTED steps; undetected steps may happen in between.
DETECT = {"IA": 1.0, "EX": 0.4, "PE": 0.3, "PR": 0.3, "DE": 0.2,
          "CA": 0.6, "DI": 0.3, "LM": 0.6, "CO": 0.3, "C2": 0.4}

# Median hours spent per step, by actor (lognormal, sigma below). Illustrative.
DWELL_MEDIAN_H = np.array([3.0, 4.0, 20.0, 6.0])
DWELL_SIGMA = 1.0

# --------------------------------------------------------------------------------------
# Transition weights (relative, row-normalised later)
# --------------------------------------------------------------------------------------
BASE = {
    "IA": {"EX": 6, "PE": 1, "DE": 1, "CA": 2},
    "EX": {"PE": 3, "PR": 2, "DE": 2, "CA": 2, "DI": 2, "C2": 3},
    "PE": {"PR": 2, "DE": 2, "CA": 2, "DI": 2, "C2": 3},
    "PR": {"DE": 2, "CA": 4, "DI": 2, "LM": 2},
    "DE": {"CA": 3, "DI": 3, "LM": 2, "PR": 1, "C2": 1},
    "CA": {"DI": 2, "LM": 5, "PR": 2, "DE": 1},
    "DI": {"LM": 4, "CA": 2, "CO": 1, "DE": 1},
    "LM": {"DI": 3, "CA": 2, "PR": 1, "CO": 1, "PE": 1, "DE": 1,
           # small residual chance that any actor ends in any objective
           "EXF": 0.05, "RAN": 0.05, "DES": 0.05, "CRED": 0.05},
    "CO": {"C2": 2, "DI": 1, "LM": 1},
    "C2": {"DI": 2, "CO": 2, "LM": 1, "EX": 1},
}

OVERLAYS = {
    "Ransomware affiliate": {
        "IA": {"EX": 2},
        "DE": {"RAN": 2.5, "LM": 1},
        "CA": {"LM": 2, "RAN": 0.3},
        "DI": {"RAN": 0.8, "LM": 1},
        "LM": {"RAN": 2.5, "DE": 1, "CO": 0.5, "EXF": 0.3},
        "CO": {"EXF": 3},
        "C2": {"RAN": 0.8, "EXF": 0.6},
    },
    "Access / credential broker": {
        "IA": {"CA": 5},
        "EX": {"CA": 3},
        "CA": {"CRED": 3.0},
        "DI": {"CA": 1, "CRED": 0.5},
        "LM": {"CRED": 1.2, "CA": 1},
        "CO": {"EXF": 1.0, "CRED": 1},
        "C2": {"CRED": 0.5},
    },
    "Espionage APT": {
        "PE": {"DE": 2, "C2": 1},
        "DE": {"DI": 1, "PE": 1},
        "CA": {"DI": 1, "CRED": 0.4},
        "DI": {"CO": 3},
        "LM": {"CO": 3, "DI": 1, "PE": 1, "EXF": 0.2},
        "CO": {"EXF": 4, "C2": 1},
        "C2": {"EXF": 1.5, "CO": 2},
    },
    "Destructive (wiper)": {
        "PR": {"DES": 0.4},
        "DE": {"DES": 1.8},
        "DI": {"DES": 0.6, "LM": 1},
        "LM": {"DES": 2.2, "PR": 1},
        "C2": {"DES": 0.6},
        "CO": {"DES": 0.5, "EXF": 0.5},
    },
}


def build_matrices():
    """Return P with shape (n_actors, n_states, n_states); rows sum to 1.
    Objective (absorbing) states transition to themselves."""
    P = np.zeros((len(ACTORS), N_STATES, N_STATES))
    for a, actor in enumerate(ACTORS):
        for src, row in BASE.items():
            for dst, w in row.items():
                P[a, IDX[src], IDX[dst]] += w
        for src, row in OVERLAYS[actor].items():
            for dst, w in row.items():
                P[a, IDX[src], IDX[dst]] += w
        P[a, :N_TACTICS] /= P[a, :N_TACTICS].sum(axis=1, keepdims=True)
        for k in range(N_TACTICS, N_STATES):
            P[a, k, k] = 1.0
    return P


BASE_P = build_matrices()


# --------------------------------------------------------------------------------------
# Exact (analytic) answers, used only to VERIFY the Monte Carlo engine
# --------------------------------------------------------------------------------------
def absorption_probs(P_actor):
    """B[i, j] = P(end in objective j | currently in tactic i), via fundamental matrix."""
    Q = P_actor[:N_TACTICS, :N_TACTICS]
    R = P_actor[:N_TACTICS, N_TACTICS:]
    N = np.linalg.inv(np.eye(N_TACTICS) - Q)
    return N @ R


def evidence_likelihood(P_actor, evidence, detect=None):
    """Exact P(the DETECTED steps so far are exactly `evidence` | actor).

    Builds an augmented chain on (tactic, #evidence-items-matched) and solves for the
    probability of absorbing into 'all evidence matched'. Starts in evidence[0].
    A step is detected with probability q[tactic]; a detected step that is not the
    next evidence item contradicts the evidence (path rejected)."""
    q = detect_vector() if detect is None else detect
    ev = [IDX[e] for e in evidence]
    L = len(ev)
    if L == 1:
        return 1.0
    # transient augmented states: (s, k) for s in tactics, k in 1..L-1
    n = N_TACTICS * (L - 1)

    def aid(s, k):
        return (k - 1) * N_TACTICS + s

    Q = np.zeros((n, n))
    r = np.zeros(n)  # one-step prob of completing the match
    for k in range(1, L):
        for s in range(N_TACTICS):
            for s2 in range(N_STATES):
                p = P_actor[s, s2]
                if p == 0:
                    continue
                if s2 >= N_TACTICS:
                    continue  # objective reached before evidence completed -> failure
                if s2 == ev[k]:  # detected -> advance
                    if k + 1 == L:
                        r[aid(s, k)] += p * q[s2]
                    else:
                        Q[aid(s, k), aid(s2, k + 1)] += p * q[s2]
                # undetected -> keep waiting for the same evidence item
                Q[aid(s, k), aid(s2, k)] += p * (1 - q[s2])
                # (detected but not the expected item -> contradicts evidence)
    h = np.linalg.solve(np.eye(n) - Q, r)
    return h[aid(ev[0], 1)]


def detect_vector():
    return np.array([DETECT[c] for c, _ in TACTICS])


def exact_prediction(P, prior, evidence):
    """Exact posterior over actors and objective distribution, for FIXED matrices."""
    like = np.array([evidence_likelihood(P[a], evidence) for a in range(len(ACTORS))])
    post = prior * like
    post /= post.sum()
    last = IDX[evidence[-1]]
    obj = sum(post[a] * absorption_probs(P[a])[last] for a in range(len(ACTORS)))
    nxt = sum(post[a] * P[a, last] for a in range(len(ACTORS)))
    return post, obj, nxt
