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

def get_index_thrust(thrust_profil):
	index = np.array([], dtype=int)
	for k, T in enumerate(thrust_profil[0]):
		if T > 1e-5:
			index = np.append(index, int(k))

	return index

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

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(propagation.y[0], propagation.y[1], '-', color='blue', linewidth=1)
	angle = 2 * np.pi * propagation.t[-1] / cst.T_M
	r_m_f = R2_6d(angle).dot(np.array([cst.d_M, 0, 0, 0, cst.V_M, 0]))
	ax.plot([r_m_f[0]], [r_m_f[1]], 'o', color='black')
	plot_env_2D(ax)

	plt.grid()
	plt.show()

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

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(trajectory[0, index:], trajectory[1, index:], '-', color='blue', linewidth=1)

	plt.grid()
	plt.show()

	return trajectory[:, index:], time[index:], trajectory[:, :index], time[:index]

def modify_last_arc(trajectory, time, Tmax, mass, eps, theta):

	# Propagation parameters
	t_span = [time[0], time[0] + 365 * 86400]
	t_eval = np.linspace(t_span[0], t_span[-1], 1000000)
	moon_approach.terminal = True

	r0 = trajectory[:, 0]

	propagation = solve_ivp(fun=thrusted_dynamics, t_span=t_span, t_eval=t_eval, y0=r0, args=(Tmax, mass, eps, theta), \
	events=(moon_approach), rtol=1e-10, atol=1e-13)

	angle = 2 * np.pi * propagation.t[-1] / cst.T_M
	r_m_f = R2_6d(angle).dot(np.array([cst.d_M, 0, 0, 0, cst.V_M, 0]))

	pos_error = np.linalg.norm( propagation.y[:3, -1] - r_m_f[:3] )
	vel_error = np.linalg.norm( propagation.y[3:, -1] - r_m_f[3:] )

	print("Position error : {} km".format(pos_error))
	print("Velocity error : {} km/s".format(vel_error))

	return propagation.y, propagation.t


def assembly(trajectory_prv, time_prv, thrust_prv, trajectory_add, time_add, thrust_add):

	eci_trajectory = np.hstack((trajectory_prv, trajectory_add))
	eci_time = np.concatenate((time_prv, time_add))
	thrusts = np.hstack((thrust_prv, thrust_add))

	thrust_index = get_index_thrust(thrusts)

	moon_angle = 2*np.pi * eci_time[-1] / cst.T_M
	r_M_f = R2_6d(moon_angle).dot(np.array([cst.d_M, 0, 0, 0, cst.V_M, 0]))

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(eci_trajectory[0], eci_trajectory[1], '-', color='blue', linewidth=1)
	ax.plot(eci_trajectory[0, thrust_index], eci_trajectory[1, thrust_index], '-', color='red', linewidth=1)
	ax.plot([r_M_f[0]], [r_M_f[1]], 'o', color='black', markersize=2)
	plot_env_2D(ax)

	plt.grid()
	plt.show()

	mu = 0.012151
	L = 384400
	T = 2360591.424
	V = L/(T/(2*np.pi))
	cr3bp = CR3BP(mu=mu, L=L, V=V, T=T/(2*np.pi))

	syn_trajectory = np.ndarray(shape=eci_trajectory.shape)
	syn_time = eci_time / cr3bp.T

	for k, t in enumerate(syn_time):
		syn_trajectory[:, k] = cr3bp.eci2syn(t, np.concatenate((eci_trajectory[:3, k]/cr3bp.L, eci_trajectory[3:, k]/cr3bp.V)) )


	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(syn_trajectory[0], syn_trajectory[1], '-', color='blue', linewidth=1)
	ax.plot([-cr3bp.mu], [0], 'o', color='black', markersize=5)
	ax.plot([1-cr3bp.mu], [0], 'o', color='black', markersize=2)

	plt.grid()
	plt.show()


def thrust_profil(trajectory_fx, time_fx, trajectory_add, time_add, eps, eps_l, theta, Tmax):

	axis = R2(theta).dot([-1, 0, 0])

	thrust_fx = np.zeros((4, time_fx.shape[0]))
	thrust_add = np.zeros((4, time_add.shape[0]))

	for k, t in enumerate(time_fx):

		gamma = np.sign( np.cross(axis, trajectory_fx[:3, k])[2] ) * np.arccos( np.dot(axis, trajectory_fx[:3, k]) / np.linalg.norm(trajectory_fx[:3, k])  )
		thrust_on = True if abs(gamma) <= eps else False

		if thrust_on == True:
			v = np.linalg.norm(trajectory_fx[3:, k])
			thrust_fx[1:, k] = trajectory_fx[3:, k] / v
			thrust_fx[0, k]  = Tmax 


	for k, t in enumerate(time_add):

		gamma = np.sign( np.cross(axis, trajectory_add[:3, k])[2] ) * np.arccos( np.dot(axis, trajectory_add[:3, k]) / np.linalg.norm(trajectory_add[:3, k])  )
		thrust_on = True if abs(gamma) <= eps_l else False

		if thrust_on == True:
			v = np.linalg.norm(trajectory_add[3:, k])
			thrust_add[1:, k] = trajectory_add[3:, k] / v
			thrust_add[0, k]  = Tmax 

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(time_fx, thrust_fx[0], color='blue')
	ax.plot(time_add, thrust_add[0], color='blue')

	plt.grid()
	plt.show()

	return thrust_fx, thrust_add


if __name__ == '__main__':

	theta = 119.62499					# Initial orbit orientation [°]
	Tmax  = 2 						    # Maximum thrust [N]
	mass  = 2000 						# S/C initial mass [kg]

	eps = 130	    					# Thrust arc semi-angle [°]
	eps_l = 11.5315315315        		# Thrust arc semi-angle on last branch [°]
	r_p = 300 	    					# Earth orbit perigee [km]
	r_a = 30000     					# Earth orbit apogee  [km]

	trajectory, time = moon_orbit_reaching(Tmax/1000, mass, r_p, r_a, eps*np.pi/180, theta*np.pi/180)

	trajectory_ut, time_ut, trajectory_fx, time_fx = keep_last_branch(trajectory, time)

	trajectory_add, time_add = modify_last_arc(trajectory_ut, time_ut, Tmax/1000, mass, eps_l*np.pi/180, theta*180/np.pi)

	thrust_fx, thrust_add = thrust_profil(trajectory_fx, time_fx, trajectory_add, time_add, eps*np.pi/180, eps_l*np.pi/180, theta*np.pi/180, Tmax/1000)

	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/30-05-2021', 'wb') as file:
		pickle.dump({'trajectory_fx': trajectory_fx, 'time_fx': time_fx, 'thrust_fx': thrust_fx, 'trajectory_add': trajectory_add, \
			'time_add': time_add, 'thrust_add': thrust_add}, file)

	# with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/30-05-2021', 'rb') as file:
	# 	results = pickle.load(file)

	# trajectory_fx, time_fx, thrust_fx, trajectory_add, time_add, thrust_add = results.values()

	# assembly(trajectory_fx, time_fx, thrust_fx, trajectory_add, time_add, thrust_add)

