import sys
import numpy as np 
import pykep as pk
import pickle

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.earth_capture import arrival_states
from scripts.earth_capture.coc import P_ECLJ2000_ECI, P_ECI_HRV, cart2sph, Rotation
from scripts.earth_capture.CR3BP_capture_trajectory import CR3BP_moon_moon
from scripts.earth_capture.utils import kepler, angle_w_Ox, plot_env_3D, cart2kep
from scripts.earth_capture.lga import first_lga
from scripts.earth_capture import constants as cst

from scripts.earth_capture.OCP_moon_moon_leg import MoonMoonLeg
from collocation.GL_V.src.optimization import Optimization


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

# Computation of the possibles LGA to capture the NEA into bounded orbit around the Earth
possibilities, time = first_lga(r=np.concatenate((r_ECI, v_ECI)), r_m=r_m, p=2, q=1)


# Find the trajectory with the lower inclinaison w.r.t the Moon's orbital plan
inclinaisons = np.array([])
for trajectory in possibilities:
	_, _, i, _, _, _ = cart2kep(trajectory[2:5], trajectory[5:], cst.mu_E)
	inclinaisons = np.append(inclinaisons, i)

min_ = inclinaisons[0]
index = 0
for k in range(len(inclinaisons)):
	if inclinaisons[k] <= min_:
		index = k
		min_ = inclinaisons[k]


r_p = possibilities[index, 2:]

# Rotation of the states so that at t=0, the Moon is lying on the Ox axis
gamma = angle_w_Ox(r_p[:3])
r_p[3:] = Rotation(-gamma).dot(r_p[3:])
r_p[:3] = Rotation(-gamma).dot(r_p[:3])

ECI_propagation = solve_ivp(fun=kepler, t_span=(time[0], time[-1]), t_eval=time, y0=r_p, rtol=1e-12, atol=1e-12)

# Conversion into the synodic frame
cr3bp, trajectory_ut, time_ut = CR3BP_moon_moon(ECI_propagation.y, ECI_propagation.t)

# Plot of the trajectory in the Synodic frame
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(trajectory_ut[0], trajectory_ut[1], trajectory_ut[2], '-', color='blue', linewidth=1)
ax.plot([-cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
ax.plot([1-cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()

# Optimization
# ------------
moon_moon_problem = MoonMoonLeg(cr3bp, mass, Tmax/1000, trajectory_ut, time_ut)

# Instantiation of the optimization
optimization = Optimization(problem=moon_moon_problem)

# Launch of the optimization
optimization.run()

opt_trajectory = optimization.results['opt_st']

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], '-', color='blue', linewidth=1)
ax.plot([ -cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
ax.plot([1-cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()