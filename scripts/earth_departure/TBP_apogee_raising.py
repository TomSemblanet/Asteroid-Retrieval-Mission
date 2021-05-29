import sys
import numpy as np
import pykep as pk 
import pickle

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.utils import cart2sph, sph2cart, cr3bp_moon_approach, kep2cart, R2, R2_6d, plot_env_2D
from scripts.earth_departure.cr3bp import CR3BP
from scripts.earth_departure import constants as cst

def moon_altitude_reached(t, r, Tmax, mass, eps, theta):
	return np.linalg.norm(r[:3]) - (cst.d_M - 3000)


def TBP_thrusted_dynamics(t, r, Tmax, mass, eps, theta):

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
	thrust_on = 1 if abs(gamma) <= eps else 0

	if thrust_on == 1:
		v_mag = np.linalg.norm(r[3:])

		vx_dot += Tmax / mass * vx / v_mag
		vy_dot += Tmax / mass * vy / v_mag
		vz_dot += Tmax / mass * vz / v_mag


	return np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])


def moon_first_shot(theta, r0, Tmax, mass, eps, t_span, t_eval):
	theta=theta[0]
	# Rotation of the initial position
	r0 = R2_6d(theta).dot(r0)

	propagation = solve_ivp(fun=TBP_thrusted_dynamics, t_span=t_span, t_eval=t_eval, y0=r0, args=(Tmax, mass, eps, theta), \
		events=(moon_altitude_reached), rtol=1e-10, atol=1e-13)

	r_M_f = R2(2*np.pi*propagation.t[-1]/cst.T_M).dot([cst.d_M, 0, 0])

	error = np.linalg.norm(propagation.y[:3, -1] - r_M_f)

	print("Theta: {}°\tError: {}km".format(theta*180/np.pi, error), flush=True)

	return error


def TBP_apogee_raising(Tmax, mass, r_p, r_a, eps, v_inf, theta):

	# 1 - Definition of the initial circular orbit and S/C initial state in the Earth inertial frame
	# ----------------------------------------------------------------------------------------------
	a  = (2*cst.R_E + r_p + r_a) / 2		 # SMA [km]
	e  = 1 - (cst.R_E + r_p) / a			 # Eccentricity [-]
	i  = 0							 	 	 # Inclinaison [rad]
	W  = 0				 				 	 # RAAN [rad]
	w  = np.pi		 		     		 	 # Perigee anomaly [rad]
	ta = -eps  					  			 # True anomaly [rad]


	# S/C initial states on its circular orbit around the Earth [km] | [km/s]
	r0 = kep2cart(a, e, i, W, w, ta, cst.mu_E)

	theta *= np.pi / 180

	t_span = [0, 365 * 86400]
	t_eval = np.linspace(t_span[0], t_span[-1], 1000000)
	moon_altitude_reached.terminal = True

	minimization = minimize(fun=moon_first_shot, x0=[theta], args=(r0, Tmax, mass, eps, t_span, t_eval), tol=1000)
	theta = minimization.x

	r0 = R2_6d(theta).dot(r0)

	propagation = solve_ivp(fun=TBP_thrusted_dynamics, t_span=t_span, t_eval=t_eval, y0=r0, args=(Tmax, mass, eps, theta), \
		events=(moon_altitude_reached), rtol=1e-10, atol=1e-13)

	r_M_f = R2_6d(2*np.pi*propagation.t[-1] / cst.T_M).dot([cst.d_M, 0, 0, 0, cst.V_M, 0])

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(propagation.y[0], propagation.y[1], '-', color='blue', linewidth=1)
	ax.plot([r_M_f[0]], [r_M_f[1]], 'o', color='black', markersize=3)
	plot_env_2D(ax)

	plt.grid()
	plt.show()


	mu = 0.012151
	L = 384400
	T = 2360591.424
	V = L/(T/(2*np.pi))
	cr3bp = CR3BP(mu=mu, L=L, V=V, T=T/(2*np.pi))

	cr3bp_time = np.zeros(shape=propagation.t.shape)
	cr3bp_trajectory = np.zeros(shape=propagation.y.shape)

	for k, t in enumerate(propagation.t):
		propagation.y[:3, k] /= cr3bp.L 
		propagation.y[3:, k] /= cr3bp.V

		cr3bp_trajectory[:, k] = cr3bp.eci2syn(t / cr3bp.T , propagation.y[:, k])
		cr3bp_time[k] = t / cr3bp.T

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(cr3bp_trajectory[0], cr3bp_trajectory[1], '-', color='blue', linewidth=1)
	ax.plot([-cr3bp.mu], [0], 'o', color='black', markersize=5)
	ax.plot([1-cr3bp.mu], [0], 'o', color='black', markersize=2)

	plt.grid()
	plt.show()

	np.loadtxt('/home/dcas/yv.gary/SEMBLANET/Asteroid-Retrieval-Mission/local/cr3bp_apogee_raising.txt')



if __name__ == '__main__':

	theta = float(sys.argv[1])

	# Spacecraft characteristics
	# --------------------------
	Tmax = 2 	 # Maximum thrust [N]
	mass = 2000  # Mass			  [kg]

	# Trajectory parameters
	# ---------------------
	eps = 120	  # Thrust arc semi-angle [°]
	r_p = 300     # Earth orbit perigee [km]
	r_a = 30000   # Earth orbit apogee  [km]

	r_m = 300	  # S/C - Moon surface minimal distance [km]

	p = 1		# Resonance parameters (Moon) [-]
	q = 1		# Resonance parameters (S/C)  [-]

	# Outter trajectory characteristics
	# ---------------------------------
	tau = 15088.095521558473									  		 # Moon departure date (MJD2000)	
	v_out = np.array([ 0.84168181508,   0.09065171796, -0.27474864627])  # Velocity at Moon departure in the ECLIPJ2000 frame [km/s]
	v_inf = np.linalg.norm(v_out)										 # Excess velocity at Moon departure [km/s]


	TBP_apogee_raising(Tmax/1000, mass, r_p, r_a, eps*np.pi/180, v_inf, theta)
