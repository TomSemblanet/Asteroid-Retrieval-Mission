import sys
import numpy as np 
import pykep as pk
import pickle

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.earth_capture import arrival_states
from scripts.earth_capture.coc import P_ECLJ2000_ECI, P_ECI_HRV, cart2sph, Rotation
from scripts.earth_capture.CR3BP_capture_trajectory import CR3BP_moon_moon
from scripts.earth_capture.utils import kepler, angle_w_Ox, plot_env_3D, plot_env_2D, cart2kep
from scripts.earth_capture.lga import first_lga
from scripts.earth_capture import constants as cst
from scripts.earth_capture.DROs import get_state, plot_DRO_state, propagate

from scripts.utils import load_bodies, load_kernels

# Pickle file containing the S/C and trajectory informations
# ----------------------------------------------------------
file_path = sys.argv[1]

with open(file_path, 'rb') as f:
	NEA_Earth = pickle.load(f)

udp = NEA_Earth['udp']
decision_vector = NEA_Earth['population'].get_x()[0]

# Pickle file containing the optimal trajectory
# ---------------------------------------------

with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/to_DRO', 'rb') as f:
	optimization = pickle.load(f)

opt_trajectory = optimization['trajectory']
opt_controls = optimization['controls']
opt_time = optimization['time']
cr3bp = optimization['cr3bp']

print(opt_time*cr3bp.T)
input()

# Target DRO
# ----------
with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/data/DROs', 'rb') as file:
	DROs = pickle.load(file)

DRO, target = get_state(DROs, Cjac=2.9328, theta=0.5)

# r0 = DRO[0, :6]

# t_span = [0, 3.45]
# t_eval = np.linspace(t_span[0], t_span[-1], 10000)

# DRO_propagation = solve_ivp(fun=cr3bp.states_derivatives, y0=r0, t_span=t_span, t_eval=t_eval, rtol=1e-10, atol=1e-13)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.plot(DRO_propagation.y[0], DRO_propagation.y[1], '-', color='magenta', alpha=0.7)
# ax.plot([1-cr3bp.mu], [0], 'o', color='black', markersize=3)
# ax.plot([-cr3bp.mu], [0], 'o', color='black', markersize=5)

# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)

# plt.grid()
# plt.show()

# Inner trajectory characteristics
# --------------------------------
tau = decision_vector[0] + decision_vector[1]					# Moon arrival date (MJD2000)	

r_in_S, r_in_E, r_in_M = arrival_states.get(file_path)		 	# S/C states relatively to the Sun, Earth and Moon at Moon arrival [m] | [m/s]
r_in_S, r_in_E, r_in_M = r_in_S/1000, r_in_E/1000, r_in_M/1000 	# S/C states relatively to the Sun, Earth and Moon at Moon arrival [km] | [km/s]


# Conversion of the S/C velocity from the ECLJ2000 frame to the ECI one
P_ECI_ECLJ2000 = np.linalg.inv(P_ECLJ2000_ECI(tau))
r_ECI = P_ECI_ECLJ2000.dot(r_in_E[:3])
v_ECI = P_ECI_ECLJ2000.dot(r_in_E[3:])

# # Conversion of the S/C states in the synodic frame
# r_syn = cr3bp.eci2syn(0, np.concatenate((r_ECI / cr3bp.L, v_ECI / cr3bp.V)))

# t_span = [0, -1*86400]
# t_eval = np.linspace(t_span[0], t_span[-1], 10000)

# inner_trajectory = solve_ivp(fun=cr3bp.states_derivatives, y0=r_syn, t_span=t_span, t_eval=t_eval, rtol=1e-10, atol=1e-13)

for k in range(len(opt_trajectory[0])):
	opt_trajectory[:-1, k] = cr3bp.syn2eci(opt_time[k], opt_trajectory[:-1, k])

DRO_time = np.linspace(0, 6.45, len(DRO))
for k in range(len(DRO_time)):
	DRO[k, :6] = cr3bp.syn2eci(DRO_time[k]+0.4, DRO[k, :6])

# Completing the trajectory 
r0 = opt_trajectory[:-1, 0].copy()
r0[:3], r0[3:] = r0[:3] * cr3bp.L, r0[3:] * cr3bp.V
t_span = [0, - 0.55 * 86400]
t_eval = np.linspace(t_span[0], t_span[-1], 10000)

propagate = solve_ivp(fun=kepler, y0=r0, t_span=t_span, t_eval=t_eval, atol=1e-13, rtol=1e-10)

# Inner trajectory
r0 = propagate.y[:, -1]
r0[3:] += np.array([0.5, 2, -0.05])

t_span = [0, - 3 * 86400]
t_eval = np.linspace(t_span[0], t_span[-1], 10000)

propagate_inner = solve_ivp(fun=kepler, y0=r0, t_span=t_span, t_eval=t_eval, atol=1e-13, rtol=1e-10)

# Thrust / Coast phases
thrust = list()
coast = list()

thrust_On = False
coast_On = True

low, upp = 0, 0

for k, t in enumerate(opt_time):

	if (thrust_On == False and opt_controls[0, k] >= 500):
		thrust_On = True
		coast_On = False

		upp = k + 1
		coast.append([low, upp])

		low = k - 1

	elif (thrust_On == True and opt_controls[0, k] < 500):
		thrust_On = False
		coast_On = True
		upp = k + 1

		thrust.append([low, upp])

		low = k - 1

	elif (k == len(opt_time)-1):
		upp = k + 1
		if opt_controls[0, k] >= 500:
			thrust.append([low, upp])
		else:
			coast.append([low, upp])

thrust = np.array(thrust)
coast = np.array(coast)



fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(propagate.y[0], propagate.y[1], propagate.y[2], '-', color='blue', alpha=0.7, linewidth=1)
for coast_arc in coast:
	ax.plot(opt_trajectory[0, coast_arc[0]:coast_arc[1]]*cr3bp.L, opt_trajectory[1,  coast_arc[0]:coast_arc[1]]*cr3bp.L, \
		opt_trajectory[2,  coast_arc[0]:coast_arc[1]]*cr3bp.L, '-', color='blue', alpha=0.7, linewidth=1)
for thrust_arc in thrust:
	ax.plot(opt_trajectory[0, thrust_arc[0]:thrust_arc[1]]*cr3bp.L, opt_trajectory[1,  thrust_arc[0]:thrust_arc[1]]*cr3bp.L, \
		opt_trajectory[2,  thrust_arc[0]:thrust_arc[1]]*cr3bp.L, '-', color='red', alpha=1, linewidth=1)

ax.plot(DRO[:, 0]*cr3bp.L, DRO[:, 1]*cr3bp.L, DRO[:, 2]*cr3bp.L, '-', color='magenta', alpha=0.7, linewidth=1, label='DRO (J = 2.9328)')

ax.plot(propagate_inner.y[0], propagate_inner.y[1], propagate_inner.y[2], '-', color='green', alpha=0.7, linewidth=1, label='Inbound trajectory')
ax.plot([propagate.y[0, -1]], [propagate.y[1, -1]], [propagate.y[2, -1]], marker='o', color=(0, 0, 1, 0), markeredgecolor='magenta', \
		markeredgewidth=1, label='Moon encounter')

ax.plot([0, 1], [0, 1], [0, 1], '-', color='red', alpha=0.7, label='Thrust arc')
ax.plot([0, 1], [0, 1], [0, 1], '-', color='blue', alpha=0.7, label='Coast arc')

plot_env_3D(ax)

ax.set_xlim(-500e3, 500e3)
ax.set_ylim(-500e3, 500e3)
ax.set_zlim(-20e3, 20e3)

ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')

plt.legend()
plt.grid()
plt.show()