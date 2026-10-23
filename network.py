"""
network.py - A small enterprise network the attacker moves through.

Data structures
  * Graph as an ADJACENCY LIST: a dict (hash map) host -> list of neighbouring hosts.
    O(1) lookup of a host's neighbours at every simulated lateral-movement step.
  * HOST attributes in a dict: role and zone, used for drawing and impact.
  * BREADTH-FIRST SEARCH with collections.deque: shortest path (in hops) from the
    attacker's compromised hosts to a target host. Multi-source BFS gives the next
    hop a targeted attacker takes toward its "crown jewel".
  * The attacker's compromised hosts are a SET (hash set): O(1) membership tests.

The layout is illustrative (24 hosts, 3 zones), not a real organisation's network.
"""

from collections import deque

# host -> (role, zone)
HOSTS = {
    **{f"WS{i:02d}": ("workstation", "user") for i in range(1, 11)},
    "MAIL1": ("server", "user"), "PRT1": ("server", "user"), "VPN1": ("server", "server"),
    "FS1": ("server", "server"), "APP1": ("server", "server"), "APP2": ("server", "server"),
    "WEB1": ("server", "server"), "JUMP1": ("server", "server"), "HR1": ("server", "server"),
    "DB1": ("crown", "core"), "DB2": ("server", "core"),
    "DC1": ("crown", "core"), "DC2": ("server", "core"), "BKP1": ("crown", "core"),
}

_EDGES = [
    # user segment: workstations in a ring, all reach mail, print and the file server
    *[(f"WS{i:02d}", f"WS{i % 10 + 1:02d}") for i in range(1, 11)],
    *[(f"WS{i:02d}", "MAIL1") for i in range(1, 11)],
    *[(f"WS{i:02d}", "PRT1") for i in (1, 3, 5, 7, 9)],
    *[(f"WS{i:02d}", "FS1") for i in (2, 4, 6, 8, 10)],
    ("WS01", "APP1"), ("WS05", "HR1"), ("WS09", "JUMP1"),
    # server segment
    ("APP1", "APP2"), ("APP1", "WEB1"), ("APP1", "JUMP1"), ("APP1", "FS1"),
    ("APP2", "DB2"), ("WEB1", "DB2"), ("WEB1", "VPN1"), ("HR1", "DB1"), ("FS1", "HR1"),
    # core: identity, data, backups
    ("JUMP1", "DC1"), ("VPN1", "DC1"), ("DC1", "DC2"), ("DB2", "DB1"),
    ("DC2", "BKP1"), ("DB1", "BKP1"),
]

# adjacency list (dict of lists)
ADJ = {h: [] for h in HOSTS}
for a, b in _EDGES:
    ADJ[a].append(b)
    ADJ[b].append(a)

FOOTHOLD = ["WS01", "APP1"]          # phished laptop + the "new server" the attacker logged in to

# each attacker type's crown jewel: reaching it makes the objective far more damaging
TARGET = {
    "Ransomware affiliate": "BKP1",        # destroy backups so victims must pay
    "Access / credential broker": "DC1",   # domain controller = every password
    "Espionage APT": "DB1",                # the customer / IP database
    "Destructive (wiper)": "DC1",          # push the wiper to every host from the DC
}


def bfs_hops(sources, adj=ADJ):
    """Multi-source BFS: hop distance from the nearest source to every host."""
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def next_hop_toward(compromised, target, adj=ADJ):
    """BFS from the compromised set; return the first uncompromised host on a
    shortest path to `target` (None if target already owned or unreachable)."""
    if target in compromised:
        return None
    start = sorted(compromised)               # sorted: set order varies between Python runs
    parent = {s: None for s in start}
    q = deque(start)
    while q:
        u = q.popleft()
        if u == target:
            break
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    if target not in parent:
        return None
    node = target
    while parent[node] not in compromised:     # walk back to the hop next to our hosts
        node = parent[node]
    return node


def frontier(compromised, adj=ADJ):
    """Uncompromised hosts adjacent to any compromised host."""
    return sorted({v for u in compromised for v in adj[u] if v not in compromised})
