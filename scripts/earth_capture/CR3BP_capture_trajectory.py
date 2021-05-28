import sys
import numpy as np
import pykep as pk 
import pickle

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.utils import cart2sph, sph2cart, cr3bp_moon_approach
from scripts.earth_departure.cr3bp import CR3BP
from scripts.earth_departure import constants as cst
from scripts.utils import load_bodies, load_kernels

from scripts.earth_departure.OCP_moon_moon_leg import MoonMoonLeg
from scripts.earth_departure.OCP_apogee_raising import ApogeeRaising
from collocation.GL_V.src.optimization import Optimization


def moon_point_approach(theta, r0, r_tgt, t_span, t_eval, cr3bp, mass, Tmax, thrusts_intervals):
	# Angle extraction
	theta = theta[0]

	# Rotation of the initial states
	r0 = states_rotation(theta).dot(r0)

	# Bring back the vector in the synodic frame
	r0 = cr3bp.eci2syn(t=0, r=r0)

	solution = solve_ivp(fun=cr3bp_dynamics_augmtd, t_span=t_span, t_eval=t_eval, y0=r0, args=(cr3bp, mass, Tmax, thrusts_intervals), \
		events=(cr3bp_moon_approach), method='LSODA', rtol=1e-13, atol=1e-13)

	return np.linalg.norm(solution.y[:3, -1] - r_tgt)

def states_rotation(theta):
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)

	Rot1 = np.hstack((np.array([[cos_theta, -sin_theta, 0],  [sin_theta, cos_theta, 0], [0, 0, 1]]), np.zeros((3, 3))))
	Rot2 = np.hstack((np.zeros((3, 3)), np.array([[cos_theta, -sin_theta, 0],  [sin_theta, cos_theta, 0], [0, 0, 1]])))
	Rot = np.vstack((Rot1, Rot2))

	return Rot



def CR3BP_moon_moon(trajectory, time):

	# 1 - Instantiation of the Earth - Moon CR3BP
	# -------------------------------------------
	mu = 0.012151
	L = 384400
	T = 2360591.424
	cr3bp = CR3BP(mu=mu, L=L, V=L/(T/(2*np.pi)), T=T/(2*np.pi))

	# Adimensionement of the ...
	# ... position
	trajectory[:3] /= cr3bp.L

	# ... velocity
	trajectory[3:] /= cr3bp.V

	# ... time
	time /= cr3bp.T 


	# 2 - Transformation from the Earth centered inertial (ECI) frame to the synodic frame
	# ------------------------------------------------------------------------------------
	for k, t in enumerate(time):
		trajectory[:, k] = cr3bp.eci2syn(t, trajectory[:, k])

	# 3 - Keep the part of the trajectory that is more than 30,000km from the Moon
	# ----------------------------------------------------------------------------
	trajectory_ut = np.empty((0, 6))
	time_ut = np.empty(0)

	r_m = np.array([1 - cr3bp.mu, 0, 0])
	dist_min = 30000 / cr3bp.L

	for k in range(len((time))):
		d = np.linalg.norm(trajectory[:3, k] - r_m)
		if d >= dist_min:
			trajectory_ut = np.vstack((trajectory_ut, trajectory[:, k]))
			time_ut = np.append(time_ut, time[k])

	trajectory_ut = np.transpose(trajectory_ut)

	return cr3bp, trajectory_ut, time_ut

