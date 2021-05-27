import sys
import numpy as np 
import pykep as pk
import pickle

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.earth_capture import arrival_states
from scripts.earth_capture.coc import P_ECLJ2000_ECI, P_ECI_HRV, cart2sph
from scripts.earth_capture.utils import C3
from scripts.earth_capture.lga import first_lga
from scripts.earth_capture import constants as cst


from scripts.utils import load_bodies, load_kernels

# Pickle file containing the S/C and trajectory informations
# ----------------------------------------------------------
file_path = sys.argv[1]

with open(file_path, 'rb') as f:
	NEA_Earth = pickle.load(f)

udp = NEA_Earth['udp']
decision_vector = NEA_Earth['population'].get_x()[0]

# Spacecraft characteristics
# --------------------------
Tmax = 2 	 # Maximum thrust [N]
mass = 6900  # Mass			  [kg]

# Trajectory parameters
# ---------------------
r_m = 300	  # S/C - Moon surface minimal distance [km]

# Outter trajectory characteristics
# ---------------------------------
tau   = decision_vector[0] + decision_vector[1]					# Moon arrival date (MJD2000)	

r_in_S, r_in_E, r_in_M = arrival_states.get(file_path)		 	# S/C states relatively to the Sun, Earth and Moon at Moon arrival [m] | [m/s]
r_in_S, r_in_E, r_in_M = r_in_S/1000, r_in_E/1000, r_in_M/1000 	# S/C states relatively to the Sun, Earth and Moon at Moon arrival [km] | [km/s]

v_inf = np.linalg.norm(r_in_M[3:])					  			# S/C excess velocity relatively to the Moon [km/s]



# Conversion of the S/C velocity from the ECLJ2000 frame to the ECI one
P_ECI_ECLJ2000 = np.linalg.inv(P_ECLJ2000_ECI(tau))
r_ECI = P_ECI_ECLJ2000.dot(r_in_E[:3])
v_ECI = P_ECI_ECLJ2000.dot(r_in_E[3:])

first_lga(r=np.concatenate((r_ECI, v_ECI)), r_m=r_m, p=2, q=1)
