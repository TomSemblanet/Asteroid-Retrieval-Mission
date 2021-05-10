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

def kepler_thrust(t, r, T, eps):
	""" Computation of the states derivatives following Keplerian mechanics.
		The S/C thruster are On when it's on an arc defined by the angles [-eps, eps] """

	# Earth's gravitational parameter [km^3/s^2]
	mu_E = 398600.4418

	x, y, z, vx, vy, vz = r

	# Earth distance [km]
	d = np.linalg.norm(r[:3])

	# S/C velocity [km/s]
	v = np.linalg.norm(r[3:])

	# Angle between the Earth and the S/C
	phi = np.arccos( np.dot( np.array([-1, 0, 0]), r[:3]) / d )

	# Thrust On/Off
	thrust_on = 1 if phi <= eps else 0

	# Position derivatives [km/s]
	x_dot = vx
	y_dot = vy
	z_dot = vz

	# Velocity derivatives [km/s^2]
	vx_dot = - mu_E / d**3 * x + thrust_on * T * vx / v
	vy_dot = - mu_E / d**3 * y + thrust_on * T * vy / v
	vz_dot = - mu_E / d**3 * z + thrust_on * T * vz / v

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

def kep2cart(a, e, i, W, w, ta, mu) : 

	h = np.sqrt(mu*a*(1-e**2))

	r_orbital_plan = (h*h/mu)*(1/(1+e*np.cos(ta)))*np.array([np.cos(ta), np.sin(ta), 0])
	v_orbital_plan = mu/h*np.array([-np.sin(ta), e+np.cos(ta), 0])


	R1 = np.array([ [np.cos(w), -np.sin(w), 0.],
			        [np.sin(w),  np.cos(w), 0.],
			        [             0.,               0., 1.] ])

	R2 = np.array([ [1.,               0.,               0.],
		            [0.,      np.cos(i),     -np.sin(i)],
		            [0.,      np.sin(i),      np.cos(i)] ])

	R3 = np.array([ [np.cos(W), -np.sin(W),     0.],
		            [np.sin(W),  np.cos(W),     0.],
		            [             0.,               0.,     1.] ])

	r = R3.dot(R2.dot(R1.dot(r_orbital_plan)))
	v = R3.dot(R2.dot(R1.dot(v_orbital_plan)))
	
	return np.concatenate((r, v))

def moon_reached(t, r, T=None, eps=None):

	# Earth - Moon distance [km]
	d_M = 385000

	return np.linalg.norm(r[:3]) - d_M
