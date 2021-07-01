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
from scripts.earth_capture.DROs import get_state, plot_DRO_state, propagate

from scripts.earth_capture.OCP_cr3bp_adaptation import MoonMoonLeg
from scripts.earth_capture.OCP_moon_moon_mass import MoonMoonLegMass
from scripts.earth_capture.OCP_moon_moon_pos import MoonMoonLegPosition
from scripts.earth_capture.OCP_moon_moon_vel import MoonMoonLegVelocity
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
Tmax = 5 	  # Maximum thrust [N]
mass = 13582  # Mass		   [kg]

# Trajectory parameters
# ---------------------
r_m = 300	  # S/C - Moon surface minimal distance [km]

# Inner trajectory characteristics
# --------------------------------
tau   = decision_vector[0] + decision_vector[1]					# Moon arrival date (MJD2000)	

r_in_S, r_in_E, r_in_M = arrival_states.get(file_path)		 	# S/C states relatively to the Sun, Earth and Moon at Moon arrival [km] | [km/s]
v_inf = np.linalg.norm(r_in_M[3:])					  			# S/C excess velocity relatively to the Moon [km/s]

print("Excess velocity w.r.t the Moon : {} km/s".format(np.linalg.norm(r_in_M[3:])))

# Target DRO
# ----------
with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/data/DROs', 'rb') as file:
	DROs = pickle.load(file)

DRO, target = get_state(DROs, Cjac=2.9328, theta=0.5)


# Conversion of the S/C velocity from the ECLJ2000 frame to the ECI one
P_ECI_ECLJ2000 = np.linalg.inv(P_ECLJ2000_ECI(tau))
r_ECI = P_ECI_ECLJ2000.dot(r_in_E[:3])
v_ECI = P_ECI_ECLJ2000.dot(r_in_E[3:])

# Computation of the possibles LGA to capture the NEA into bounded orbit around the Earth
possibilities, time = first_lga(r=np.concatenate((r_ECI, v_ECI)), r_m=r_m, p=1, q=1)


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


# Adaptation of the trajectory to the CR3BP dynamics
# --------------------------------------------------
moon_moon_problem = MoonMoonLeg(cr3bp, mass, Tmax/1000, trajectory_ut, time_ut)

# Instantiation of the optimization
optimization = Optimization(problem=moon_moon_problem)

# Launch of the optimization
optimization.run()

opt_trajectory = optimization.results['opt_st']

fig = plt.figure()
ax = fig.gca(projection='3d')

for k, Cjac in enumerate(DROs.keys()):
	if 	k % 10 == 0:
		ax.plot(DROs[Cjac][:, 0], DROs[Cjac][:, 1], '-', color='orange', linewidth=1)

ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], '-', color='blue', linewidth=1)
ax.plot([ -cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
ax.plot([1-cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()


# Optimization on the position
# ----------------------------
moon_moon_position_problem = MoonMoonLegPosition(cr3bp, mass, Tmax/1000, trajectory_ut, time_ut, target)

# Instantiation of the optimization
optimization = Optimization(problem=moon_moon_position_problem)

# Launch of the optimization
optimization.run()

opt_trajectory = optimization.results['opt_st']
opt_controls = optimization.results['opt_ct']
opt_time = optimization.results['opt_tm']

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], '-', color='blue', linewidth=1)
ax.plot(DRO[:, 0], DRO[:, 1], DRO[:, 2], '-', color='orange', linewidth=1)
ax.plot([target[0]], [target[1]], [target[2]], 'o', color='magenta', markersize=2)
ax.plot([ -cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
ax.plot([1-cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()



# Optimization on the velocity
# ----------------------------
moon_moon_velocity_problem = MoonMoonLegVelocity(cr3bp, mass, Tmax/1000, opt_trajectory, opt_controls, opt_time, target)

# Instantiation of the optimization
optimization = Optimization(problem=moon_moon_velocity_problem)

# Launch of the optimization
optimization.run()

opt_trajectory = optimization.results['opt_st']
opt_controls = optimization.results['opt_ct']
opt_time = optimization.results['opt_tm']

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], '-', color='blue', linewidth=1)
ax.plot(DRO[:, 0], DRO[:, 1], DRO[:, 2], '-', color='orange', linewidth=1)
ax.plot([target[0]], [target[1]], [target[2]], 'o', color='magenta', markersize=2)
ax.plot([ -cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
ax.plot([1-cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()



# Optimization
# ------------
moon_moon_mass_problem = MoonMoonLegMass(cr3bp, mass, Tmax/1000, opt_trajectory, opt_controls, opt_time)

# Instantiation of the optimization
optimization = Optimization(problem=moon_moon_mass_problem)

# Launch of the optimization
optimization.run()

opt_trajectory = optimization.results['opt_st']
opt_controls = optimization.results['opt_ct']
opt_time = optimization.results['opt_tm']

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], '-', color='blue', linewidth=1)
ax.plot([ -cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
ax.plot([1-cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()

with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/to_DRO', 'wb') as file:
	pickle.dump({'trajectory': opt_trajectory, 'time': opt_time, 'controls': opt_controls, 'cr3bp': cr3bp}, file)
