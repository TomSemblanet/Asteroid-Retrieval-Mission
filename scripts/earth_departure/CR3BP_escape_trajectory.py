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

def cr3bp_dynamics_augmtd(t, r, cr3bp, mass, Tmax, thrusts_intervals=None):

	# States extraction
	x, y, z, vx, vy, vz = r

	# Computation of states derivatives following cr3bp dynamics
	x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot = cr3bp.states_derivatives(t, r)

	# Thrust if the S/C is on a thrust arc
	k = np.searchsorted(a=thrusts_intervals[:, 0], v=t, side='right')
	k -= 1

	if thrusts_intervals[k, 1] >= t:
		v = np.linalg.norm([vx, vy, vz])
		vx_dot += Tmax * vx / v / mass
		vy_dot += Tmax * vy / v / mass
		vz_dot += Tmax * vz / v / mass

	return np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])

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


def CR3BP_orbit_raising(trajectory, time, t_last_ap_pass, thrusts_intervals, mass, Tmax):

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
	t_last_ap_pass /= cr3bp.T
	thrusts_intervals /= cr3bp.T

	# ... thrust
	Tmax /= cr3bp.L / cr3bp.T**2

	# 2 - Transformation from the Earth centered inertial (ECI) frame to the synodic frame
	# ------------------------------------------------------------------------------------
	for k, t in enumerate(time):
		trajectory[:, k] = cr3bp.eci2syn(t, trajectory[:, k])


	# 3 - Find the point of the Keplerian trajectory at 30,000km of the Moon
	# ----------------------------------------------------------------------
	index = 0

	r_m = np.array([1 - cr3bp.mu, 0, 0])
	dist_min = 30000 / cr3bp.L

	for k in range(len(time)):
		if np.linalg.norm(trajectory[:3, -(k+1)] - r_m) >= dist_min:
			index = len(time) - (k+1)
			break

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(trajectory[0, :index], trajectory[1, :index], '-', color='blue', linewidth=1)


	# 4 - Rotation of the initial states to approach the Moon correctly
	# -----------------------------------------------------------------

	t_span = np.array([time[0], time[-1]])
	t_eval = np.linspace(t_span[0], t_span[-1], 100000)
	r0 = trajectory[:, 0]

	cr3bp_moon_approach.terminal = True

	# Conversion of the initial state in the ECI frame
	r0 = cr3bp.syn2eci(t=time[0], r=r0)
	r_tgt = trajectory[:3, index]

	opt_results = minimize(fun=moon_point_approach, x0=[0], args=(r0, r_tgt, t_span, t_eval, cr3bp, mass, Tmax, thrusts_intervals), \
			tol=1e-2)
	theta_opt = opt_results.x[0]

	# Rotation of the initial state w/ the good angle and conversion in the synodic frame
	r0 = states_rotation(theta_opt).dot(r0)
	r0 = cr3bp.eci2syn(t=0, r=r0)


	# 5 - Propagation with the CR3BP equations
	# ----------------------------------------

	solution = solve_ivp(fun=cr3bp_dynamics_augmtd, t_span=t_span, t_eval=t_eval, y0=r0, args=(cr3bp, mass, Tmax, thrusts_intervals), \
		events=(cr3bp_moon_approach), method='LSODA', rtol=1e-13, atol=1e-13)
	cr3bp_trajectory = solution.y
	cr3bp_time = solution.t


	# 6 - Keep the part of the trajectory between the last apogee pass and the Moon
	# -----------------------------------------------------------------------------

	trajectory_ut = np.empty((0, 6))
	time_ut = np.empty(0)

	r_m = np.array([1 - cr3bp.mu, 0, 0])
	dist_min = 30000 / cr3bp.L

	# Find the index of the last apogee pass
	index_l_pass = np.searchsorted(cr3bp_time, t_last_ap_pass, side='right') - 1

	for k in range(len((cr3bp_time[index_l_pass:]))):
		d = np.linalg.norm(cr3bp_trajectory[:3, k] - r_m)
		if d >= dist_min:
			trajectory_ut = np.vstack((trajectory_ut, cr3bp_trajectory[:, index_l_pass+k]))
			time_ut = np.append(time_ut, cr3bp_time[index_l_pass+k])

	trajectory_ut = np.transpose(trajectory_ut)

	ax.plot(trajectory_ut[0], trajectory_ut[1], ':', color='green', linewidth=1)


	ax.plot([ -cr3bp.mu], [0], 'o', color='black', markersize=5)
	ax.plot([1-cr3bp.mu], [0], 'o', color='black', markersize=2)

	ax.set_xlabel('X [-]')
	ax.set_ylabel('Y [-]')

	plt.grid()
	plt.show()

	return cr3bp, trajectory_ut, time_ut 



# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * 



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


if __name__ == '__main__':

	arc = sys.argv[1]

	if arc == 'ar':
		with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/a_r', 'rb') as f:
			res = pickle.load(f)

		# Optimization of the Apogee Raising last branch
		cr3bp, trajectory, time = CR3BP_orbit_raising(trajectory=res['trajectory'], time=res['time'], thrusts_intervals=res['thrusts_intervals'], \
			mass=res['mass'], Tmax=res['Tmax'], t_last_ap_pass=res['last_apogee_pass_time'])
		mass0 = res['mass']
		Tmax  = res['Tmax']

		options = {'linear_solver': 'mumps'}

		problem = ApogeeRaising(cr3bp, mass0, Tmax, trajectory, time)

		# Instantiation of the optimization
		optimization = Optimization(problem=problem, **options)

		# Launch of the optimization
		optimization.run()

		opt_trajectory = optimization.results['opt_st']

		fig = plt.figure()
		ax = fig.gca(projection='3d')

		ax.plot(trajectory[0], trajectory[1], trajectory[2], ':', color='blue', linewidth=1)
		ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], '-', color='orange', linewidth=1)
		ax.plot([ -cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
		ax.plot([1-cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

		plt.show()


	elif arc == 'mm':
		with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/moon_moon', 'rb') as f:
			res = pickle.load(f)

		# Optimization of the Moon-Moon leg
		cr3bp, trajectory, time = CR3BP_moon_moon(trajectory=res['trajectory'], time=res['time'])
		mass0 = res['mass']
		Tmax  = res['Tmax']

		options = {'linear_solver': 'mumps'}

		problem = MoonMoonLeg(cr3bp, mass0, Tmax, trajectory, time)

		# Instantiation of the optimization
		optimization = Optimization(problem=problem, **options)

		# Launch of the optimization
		optimization.run()

		opt_trajectory = optimization.results['opt_st']

		fig = plt.figure()
		ax = fig.gca(projection='3d')

		ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], '-', color='orange', linewidth=1)
		ax.plot([ -cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
		ax.plot([1-cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

		plt.show()


	else:
		print("Error")
