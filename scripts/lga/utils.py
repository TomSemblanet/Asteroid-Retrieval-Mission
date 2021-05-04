import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.lga.cr3bp import CR3BP
from scripts.lga.coc import EM2SE, SE2EM

# Position of the Earth in the Sun-Earth synodic frame
r_Earth = np.array([1-3.036e-6, 0, 0])

# Earth - Moon distance scaled
R_M  = 384402 / 1.496e8

def terminaison_con(t, y):

	# Computation of the distance to the Earth
	earth_dist = np.linalg.norm(y[0:3]-r_Earth)

	if t > 0.0001:
		return earth_dist - R_M
	else:
		return 1

def initial_states(cr3bp, v_inf, alpha, theta_s0):
	""" Computation of the initial states of the S/C in the Sun-Earth synodic 
		frame """

	# Earth-Moon distance [km]
	R_M = 384402

	# Moon orbital velocity [km/s]
	V_M = 1.022

	# Computation of the states in the Earth-Moon frame 
	# position : [km] - velocity : [km/s]
	x = R_M
	y = 0
	z = 0

	vx = v_inf * np.sin(alpha)
	vy = V_M + v_inf * np.cos(alpha)
	vz = 0

	# Scaling of the variables
	x /= cr3bp.L
	y /= cr3bp.L
	z /= cr3bp.L

	vx /= cr3bp.V 
	vy /= cr3bp.V 
	vz /= cr3bp.V

	r = np.array([x, y, z, vx, vy, vz])

	# Conversion of the states in the Sun-Earth synodic frame
	r_SE = EM2SE(cr3bp, r, theta_s0)

	return r_SE

def moon_rot_matrix(rot_angle):

	return np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0,                 0,                      0, 0], 
				     [np.sin(rot_angle),  np.cos(rot_angle), 0,                 0,                      0, 0],
				     [                0,                  0, 1,                 0,                      0, 0],
				     [                0,                  0, 0, np.cos(rot_angle), -np.sin(rot_angle), 0],
				     [                0,                  0, 0, np.sin(rot_angle),  np.cos(rot_angle), 0],
				     [                0,                  0, 0,                 0,                      0, 1]])

def plot_env(ax, cr3bp, theta_s0):

	# Construction of the Sun representation
	sun_r = np.array([-cr3bp.mu, 0, 0, 0, 0, 0]) # In the Sun-Earth frame
	sun_r_EM = SE2EM(cr3bp, sun_r, theta_s0) # In the Earth-Moon frame

	# Construction of the Moon orbit
	x = np.linspace(-384402/cr3bp.L, 384402/cr3bp.L, 1000)
	y_p = np.zeros(1000)
	y_m = np.zeros(1000)
	for i, x_ in enumerate(x):
		y_p[i] =  np.sqrt((384402/cr3bp.L)**2 - x_**2)
		y_m[i] = -np.sqrt((384402/cr3bp.L)**2 - x_**2)

	# Plot of the Moon orbit
	ax.plot(x, y_p, '--', color='grey')
	ax.plot(x, y_m, '--', color='grey')

	ax.plot([sun_r_EM[0]/100], [sun_r_EM[1]/100], 'o', color='yellow', markersize=9, label='Sun')
	ax.plot([0], [0], 'o', color='black', markersize=9, label='Earth')
	ax.plot([384402/cr3bp.L], [0], 'o', color='black', markersize=4, label='Moon at t0')

	ax.set_xlim(-0.011, 0.011)
	ax.set_ylim(-0.011, 0.011)

	plt.grid()
	plt.legend()
	plt.show()




def secant_mtd_f(cr3bp, v_inf, alpha, theta_s0, t_span):

	# Initial states in the Sun-Earth synodic frame
	r0 = initial_states(cr3bp, v_inf, alpha, theta_s0)

	# Propagation of the CR3BP equations
	sol = solve_ivp(fun=cr3bp.states_derivatives, t_span=t_span, y0=r0, method='RK45', events=terminaison_con, rtol=1e-13, atol=1e-12)

	# Convertion of the final states in the Earth-Moon frame
	r_f = SE2EM(cr3bp, sol.y[:, -1], theta_s0)
	t_f = sol.t[-1]

	# Computation of the Moon position at t=tf
	n_per = t_f*cr3bp.T / 2551392
	rotation_angle = 2 * np.pi * n_per
	r_M_i = r_M_i = np.array([384402/cr3bp.L, 0, 0, 0, 1.0220/cr3bp.V, 0])
	r_M_f = moon_rot_matrix(rotation_angle).dot(r_M_i)

	# Computation of f(alpha)
	r_f_unt = r_f[0:3] / np.linalg.norm(r_f[0:3])
	r_M_unt = r_M_f[0:3] / np.linalg.norm(r_M_f[0:3])

	return np.sign(np.cross(r_M_unt, r_f_unt)[2]) * np.arccos(r_M_unt.dot(r_f_unt))

