import sys
import numpy as np
import pykep as pk 
import pickle

import matplotlib.pyplot as plt

from mpi4py import MPI

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.utils import cart2sph, sph2cart, cr3bp_moon_approach, kep2cart, R2, R2_6d, plot_env_2D
from scripts.earth_departure.cr3bp import CR3BP
from scripts.earth_departure import constants as cst


def moon_approach(t, r, Tmax, mass, eps, theta):
	""" Raises an event if the S/C approach the Moon's orbit at less than ``dist`` km """
	dist = 10000 # Minimal distance [km]
	return np.linalg.norm(r[:3]) - (cst.d_M - dist)

def thrusted_dynamics(t, r, Tmax, mass, eps, theta):
	""" Returns the state derivatives of a thrusted S/C in the CR3BP dynamic environment """

	x, y, z, vx, vy, vz = r 

	# Moon's position and velocity [km] | [km/s]
	r_M = np.array([cst.d_M, 0, 0, 0, cst.V_M, 0])
	r_M = R2_6d(2*np.pi * t / cst.T_M).dot(r_M)

	# S/C distance to the Earth and Moon [km]
	d_E = np.linalg.norm(r[:3])
	d_M = np.linalg.norm(r[:3] - r_M[:3])

	# S/C states derivatives
	x_dot = vx
	y_dot = vy
	z_dot = vz
	vx_dot = - cst.mu_E / d_E**3 * x - cst.mu_M / d_M**3 * (x - r_M[0]) 
	vy_dot = - cst.mu_E / d_E**3 * y - cst.mu_M / d_M**3 * (y - r_M[1])
	vz_dot = - cst.mu_E / d_E**3 * z - cst.mu_M / d_M**3 * (z - r_M[2])

	axis = R2(theta).dot([-1, 0, 0])

	gamma = np.sign( np.cross(axis, r[:3])[2] ) * np.arccos( np.dot(axis, r[:3]) / d_E  )
	thrust_on = True if abs(gamma) <= eps else False

	if thrust_on == True:
		v_mag = np.linalg.norm(r[3:])

		vx_dot += Tmax / mass * vx / v_mag
		vy_dot += Tmax / mass * vy / v_mag
		vz_dot += Tmax / mass * vz / v_mag


	return np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])


def moon_orbit_reaching(Tmax, mass, r_p, r_a, eps, theta):
	""" Constructs a trajectory departing from a ``r_p`` x ``r_a`` orbit around the Earth and arriving 
		at a given distance of the Moon's orbit """

	# 1 - Definition of the initial orbit and S/C initial state in the Earth inertial frame
	a  = (2*cst.R_E + r_p + r_a) / 2		 # SMA [km]
	e  = 1 - (cst.R_E + r_p) / a			 # Eccentricity [-]
	i  = 0							 	 	 # Inclinaison [rad]
	W  = 0				 				 	 # RAAN [rad]
	w  = np.pi		 		     		 	 # Perigee anomaly [rad]
	ta = -eps  					  			 # True anomaly [rad]

	r0 = kep2cart(a, e, i, W, w, ta, cst.mu_E)


	# 2 - Propagation parameters
	t_span = [0, 365 * 86400]
	t_eval = np.linspace(t_span[0], t_span[-1], 1000000)
	moon_approach.terminal = True

	r0 = R2_6d(theta).dot(r0) # Rotation of the initial state

	propagation = solve_ivp(fun=thrusted_dynamics, t_span=t_span, t_eval=t_eval, y0=r0, args=(Tmax, mass, eps, theta), \
		events=(moon_approach), rtol=1e-10, atol=1e-13)

	trajectory, time = propagation.y, propagation.t

	return trajectory, time

def keep_last_branch(trajectory, time):

	index = -1

	for k in range(len(time)):
		d_k_p = np.linalg.norm(trajectory[:3,     -1-k])
		d_k   = np.linalg.norm(trajectory[:3, -1-(k+1)])
		d_k_m = np.linalg.norm(trajectory[:3, -1-(k+2)])

		if (d_k_p - d_k <= 0 and d_k - d_k_m <= 0):
			index = -1-(2*k+1)
			break

	return trajectory[:, index:], time[index:], trajectory[:, :index], time[:index]

def modify_last_arc(trajectory, time, Tmax, mass, theta):

	error_matrix = np.zeros((2000, 2))

	eps_list = np.linspace(0, np.pi, 2000)

	# Propagation parameters
	t_span = [time[0], time[0] + 365 * 86400]
	t_eval = np.linspace(t_span[0], t_span[-1], 1000000)
	moon_approach.terminal = True

	r0 = trajectory[:, 0]

	# for k, eps in enumerate(eps_list):
	propagation = solve_ivp(fun=thrusted_dynamics, t_span=t_span, t_eval=t_eval, y0=r0, args=(Tmax, mass, eps, theta), \
	events=(moon_approach), rtol=1e-10, atol=1e-13)

	angle = 2 * np.pi * propagation.t[-1] / cst.T_M
	r_m_f = R2_6d(angle).dot(np.array([cst.d_M, 0, 0, 0, cst.V_M, 0]))

	pos_error = np.linalg.norm( propagation.y[:3, -1] - r_m_f[:3] )
	vel_error = np.linalg.norm( propagation.y[3:, -1] - r_m_f[3:] )

	error_matrix[k, 0] = pos_error
	error_matrix[k, 1] = vel_error

	return error_matrix


if __name__ == '__main__':

	theta_I = float(sys.argv[1]) 
	step    = float(sys.argv[2])

	# Creation of the communicator 
	comm = MPI.COMM_WORLD
	rank = comm.rank


	theta = theta_I + rank * step		# Initial orbit orientation [°]
	Tmax  = 2 						    # Maximum thrust [N]
	mass  = 2000 						# S/C initial mass [kg]

	eps = 130	    					# Thrust arc semi-angle [°]
	r_p = 300 	    					# Earth orbit perigee [km]
	r_a = 30000     					# Earth orbit apogee  [km]

	trajectory, time = moon_orbit_reaching(Tmax/1000, mass, r_p, r_a, eps*np.pi/180, theta*np.pi/180)

	trajectory_ut, time_ut, trajectory_fx, time_fx = keep_last_branch(trajectory, time)

	error_matrix = modify_last_arc(trajectory_ut, time_ut, Tmax/1000, mass, theta*180/np.pi)

	with open('/home/dcas/yv.gary/SEMBLANET/Asteroid-Retrieval-Mission/local/error_matrices/' + str(theta), 'wb') as file:
		pickle.dump(error_matrix, file)
