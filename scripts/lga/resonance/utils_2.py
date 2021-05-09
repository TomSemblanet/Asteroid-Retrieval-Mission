import matplotlib.pyplot as plt
import numpy as np 

def kepler(t, r):
	""" Computation of the states derivatives following Keplerian mechanics """

	# Earth's gravitational parameter [km^3/s^2]
	mu_E = 398600.4418

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

def R1():
	""" Rotation from the basis with axis following : 
			i : Moon's angular momentum
			j : Moon's position r/ Earth
			k : Moon's velocity r/ Earth

				to the basis with axis following :

			i : Moon's position r/ Earth
			j : Moon's velocity r/ Earth
			k : Moon's angular momentum
	"""

	return np.array([[0, 1, 0],
					 [0, 0, 1],
					 [1, 0, 0]])

def R2(theta):
	""" Rotation from the basis with axis following:
			i : Moon's position r/ Earth
			j : Moon's velocity r/ Earth
			k : Moon's angular momentum

				to the basis with axis following :

			i : Moon's position at t0
			j : completing the right-hand rule
			k : Moon's angular momentum
	"""

	return np.array([ [np.cos(theta), -np.sin(theta), 0],
					  [np.sin(theta),  np.cos(theta), 0],
					  [            0,              0, 1] ])


if __name__ == '__main__':

	from scipy.integrate import solve_ivp

	t_span = [0, 0.99*2360591.5104]
	r0 = np.array([385000, 0, 0, 0, 1.0150947090432465, 0])

	sol = solve_ivp(fun=kepler, t_span=t_span, y0=r0, atol=1e-12, rtol=1e-13)

	r = sol.y

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot(r[0], r[1], r[2])

	plt.show()
