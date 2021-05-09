import matplotlib.pyplot as plt
import numpy as np 

from scipy.integrate import solve_ivp

def kepler(t, r):
	""" Computation of the states derivatives following Keplerian mechanics """

	x, y, z, vx, vy, vz = r

	# Earth distance [km]
	d = np.linalg.norm(r[:3])

	# Position derivatives [km/s]
	x_dot = vx
	y_dot = vy
	z_dot = vz

	# Velocity derivatives [km/s^2]
	vx_dot = - mu_E / d**3 * x
	vy_dot = - mu_E / d**3 * y
	vz_dot = - mu_E / d**3 * z

	return np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])

