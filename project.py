#Import libraries
import numpy as np
import matplotlib as matlib

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


#User Workstation -> Admin Workstation -> Domain Controller
network_environment = {
    "user_account":['user_workstation', 'user_email'],
    "user_workstation": ['admin_workstation'],
    "admin_workstation": ["domain_controller"]
}







def simulation():
    pass
