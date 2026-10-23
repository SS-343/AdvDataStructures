#Import libraries
import numpy as np
import matplotlib as matlib
from dataclasses import dataclass

#Set Simulation Count
simulation_count = 100,000

#MITRE KILL CHAIN
"""
Reconnaissance – attacker researches the target (e.g., scanning, OSINT).
Weaponization – creating a malicious payload.
Delivery – sending the payload (phishing, USB, exposed service).
Exploitation – triggering a vulnerability.
Installation – establishing a foothold (malware, backdoor).
Command & Control (C2) – enabling remote control.
Actions on Objectives – exfiltration, ransomware, data destruction.
"""


@dataclass
class Asset:
    name: str
    compromised: bool = False
    contained: bool = False


class Environment:
    def __init__(self):
        self.hour = 0
        self.assets = {
            "user_account": Asset("UserAccount"),
            "user_workstation": Asset("UserWorkstation"),
            "user_network": Asset("UserNetwork"),
            "file_server": Asset("FileServer"),
            "workstation_with_admin": Asset("AdminWorkstation"),
            "domain_controller": Asset("DomainController")
        }

        self.connections = {
        "user_account":['user_workstation'],
        "user_workstation": ['user_network'],
        "user_network": ['file_server', 'workstation_with_admin'],
        "workstation_with_admin": ["domain_controller"]
        }

    def compromise(self,name):
        asset = self.assets[name]
        if not asset.contained:
            asset.compromised = True

    def contain(self,name):
        self.assets[name].contained = True
    



malware_profiles = {
    "ransomware": {
        "spread": 0.85,
        "contain": 0.15
    }
}

def simulation():
    pass
