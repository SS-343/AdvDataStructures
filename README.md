# Predicting an attacker's next move with Monte Carlo simulation

An intrusion is caught halfway through: a phishing email was opened, a stolen password
was used, and someone logged in to a new server. Where is the attacker going next?
This project simulates millions of plausible intrusions and keeps only the ones that
match what the SOC has seen. That turns the clues into probabilities for the
attacker's next step, final objective, and time remaining.

**Project topic (cybersecurity): Threat Modeling.** We model the lifecycle of a threat
to see how different types of attack progress through a network over time. We also
extend the model into risk assessment and security investment analysis using a Loss
Exceedance Curve.

## Research question
Given an observed sequence of MITRE ATT&CK tactics in an enterprise intrusion, what is
the probability distribution over the attacker's next tactic and final objective
(exfiltration, ransomware, destruction, credential theft)? How sensitive is that
distribution to uncertainty about threat-actor type and transition probabilities?

## Files
| File | What it does |
|---|---|
| `model.py` | States (10 ATT&CK tactics + 4 objectives), 4 threat-actor profiles, transition matrices, detection rates, dwell times, and the exact Markov-chain solution used for verification |
| `simulate.py` | Monte Carlo engine: vectorised random walks, rejection sampling on the evidence, and nested sampling of the transition matrices from a Dirichlet distribution |
| `network.py` | A 24-host enterprise network as an adjacency list (dict of lists), with breadth-first search (`collections.deque`) for shortest paths to the crown jewels |
| `impact.py` | Discrete-event simulation of the rest of the incident. A `heapq` priority queue races attacker steps against containment by the security team, and financial losses are drawn from lognormal distributions calibrated to the IBM Cost of a Data Breach Report 2025 |
| `experiments.py` | Runs every experiment and writes `results/results.json` plus the charts |
| `predict.py` | Command-line prediction for any observed tactic sequence |

## Run it
```bash
pip install numpy matplotlib
python experiments.py                 # about 2 minutes, writes results/
python predict.py IA CA LM            # the slide scenario
python predict.py IA CA LM CO --prior "Defense contractor (APT-heavy)"
```
Tactic codes: `IA` Initial Access, `EX` Execution, `PE` Persistence, `PR` Privilege
Escalation, `DE` Defense Evasion, `CA` Credential Access, `DI` Discovery, `LM` Lateral
Movement, `CO` Collection, `C2` Command & Control.

## Method in one paragraph
Each actor type is an absorbing Markov chain over ATT&CK tactics. The **outer loop**
draws a plausible set of transition matrices (`row ~ Dirichlet(kappa * p)`), which
represents our uncertainty about the probabilities themselves. The **inner loop**
simulates complete intrusions. Each one draws an actor from the prior, random-walks
from Initial Access to an objective, adds lognormal dwell time per step, and has each
step detected with a tactic-specific probability. An intrusion is accepted only if its
detected steps are exactly the observed evidence (rejection sampling, i.e. Bayesian
conditioning). Accepted runs give the posterior over actors, the next step, the final
objective, and time-to-objective. The spread across outer draws gives 90% intervals.

## Financial loss (E5)
Starting from the third clue, each simulated incident draws the attacker type from the
exact posterior. It then runs a discrete-event simulation:
- The earliest event in a min-heap always happens next. Events are either the
  attacker's next step or the security team containing the intrusion.
- Lateral movement adds a host to the attacker's compromised set. Moves follow the BFS
  shortest path to that attacker's crown jewel 60% of the time; otherwise the attacker
  moves to a random neighbouring host.
- The loss is drawn from a lognormal distribution. Mean losses come from IBM 2025
  where IBM gives a figure: $4.44M for a data breach and $5.08M for ransomware or
  extortion. Destruction ($5.08M), credential theft ($1.11M) and a contained incident
  ($0.25M) are assumptions. The loss is ×1.3 if the crown jewel was reached and ×0.7
  if it was not.

Comparing three response speeds turns the Loss Exceedance Curve into a security
investment comparison.

## Data structures
| Structure | Where | Why |
|---|---|---|
| NumPy arrays, vectorised | `simulate.py` | Steps every live simulated attack at once |
| PRNG (`numpy.random.default_rng`, PCG64, seeded) | everywhere | Fast, reproducible random samples |
| Hash map / adjacency list | `network.py` | O(1) neighbour lookup at each lateral move |
| Hash set | `impact.py` | O(1) "is this host already compromised?" check |
| Queue + BFS | `network.py` | Shortest path from compromised hosts to a target |
| Min-heap priority queue | `impact.py` | Discrete-event scheduling: next event in time order |

## Verification
With parameter uncertainty switched off, the simulated objective probabilities match
the exact answer from linear algebra (fundamental matrix plus an augmented evidence
chain) to within 0.2 percentage points on 1,000,000 simulated intrusions.

## Limitations
The transition weights, priors, detection rates, and dwell times are illustrative
estimates, not values fitted to incident data. The sensitivity analysis shows how much
the conclusions depend on them. Next step: estimate the matrices from labelled
incident timelines, such as MITRE ATT&CK campaign data, CISA advisories, or an
organisation's own case history.
