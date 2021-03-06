import matplotlib.pyplot as plt
import numpy as np 
import math as mt

from scripts.lga.resonance import constants as cst

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
	vx_dot = - cst.mu_E / d**3 * x
	vy_dot = - cst.mu_E / d**3 * y
	vz_dot = - cst.mu_E / d**3 * z

	return np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])

def kepler_thrust(t, r, T, eps):
	""" Computation of the states derivatives following Keplerian mechanics.
		The S/C thruster are On when it's on an arc defined by the angles [-eps, eps] """

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
	vx_dot = - cst.mu_E / d**3 * x + thrust_on * T * vx / v
	vy_dot = - cst.mu_E / d**3 * y + thrust_on * T * vy / v
	vz_dot = - cst.mu_E / d**3 * z + thrust_on * T * vz / v

	return np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])

def cart2sph(r):
	""" Converts cartesian coordinates to spherical coordinates """
	rho = np.linalg.norm(r)
	phi = mt.atan2(r[1], r[0])
	theta = theta = mt.acos(r[2]/rho)

	return np.array([rho, phi, theta])


def sph2cart(r):
	""" Converts spherical coordinates to cartesian coordinates """
	return np.array([r[0]*np.cos(r[1])*np.sin(r[2]),
				     r[0]*np.sin(r[1])*np.sin(r[2]),
				     r[0]*np.cos(r[2])])


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


def P_GEO2RVH(gamma):
	""" Passage matrix from Earth inertial frame to RVH frame """
	return np.array([[np.cos(gamma), -np.sin(gamma), 0],
		 			 [np.sin(gamma),  np.cos(gamma), 0],
		 			 [            0,              0, 1]])	

def P_RVH2GEO(gamma):
	""" Passage matrix from RVH frame to Earth inertial frame """
	return np.array([[ np.cos(gamma),  np.sin(gamma), 0],
		 			 [-np.sin(gamma),  np.cos(gamma), 0],
		 			 [            0,              0, 1]])


def P_RVH2HRV():
	""" Passage matrix from RVH frame to HRV frame """
	return np.array([[0, 1, 0],
		             [0, 0, 1],
		             [1, 0, 0]])

def P_HRV2RVH():
	""" Passage matrix from HRV frame to RVH frame """
	return np.array([[0, 0, 1],
		             [1, 0, 0],
		             [0, 1, 0]])

def P_GEO2HRV(gamma):
	""" Passage matrix from Earth inertial frame to HRV frame """
	return P_GEO2RVH(gamma).dot(P_RVH2HRV())

def P_HRV2GEO(gamma):
	""" Passage matrix from HRV frame to Earth inertial frame """
	return P_HRV2RVH().dot(P_RVH2GEO(gamma))

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


def cart2kep (r, v, mu) : 

	r_mag = np.linalg.norm(r)
	v_mag = np.linalg.norm(v)
	w_mag = np.dot(r, v)/r_mag

	h = np.cross(r, v)
	h_mag = np.linalg.norm(h)

	i = np.arccos(h[2]/h_mag)

	n = np.cross([0, 0, 1], h)
	n_mag = np.linalg.norm(n)

	if(n_mag != 0) : 
		W = np.arccos(n[0]/n_mag)
		if(n[1] < 0) : 
			W = 2*np.pi - W
	else : 
		W = 0

	e = 1/mu*((v_mag*v_mag - mu/r_mag)*r - r_mag*w_mag*v)
	e_mag = np.linalg.norm(e)

	if(n_mag != 0) : 
		if(e_mag > 1e-5) : 
			w = np.arccos(np.dot(n, e)/(n_mag*e_mag))
			if(e[2] < 0) : 
				w = 2*np.pi - w
		else : 
			w = 0
	else : 
		w = (mt.atan2(e[1], e[0]))%(2*np.pi)
		if(np.sign(np.cross(r,v))[2] < 0) : 
			w = 2*np.pi - w

	if(e_mag > 1e-5) : 
		ta = np.arccos(np.dot(e, r)/(e_mag*r_mag))
		if(w_mag < 0) : 
			ta = 2*np.pi - ta
	else : 
		if(n_mag != 0) :
			ta = np.arccos(np.dot(n, r)/(n_mag*r_mag))
			if(np.cross(n, r)[2] < 0) : 
				ta = 2*np.pi - ta
		else : 
			ta = np.arccos(r[0]/r_mag)
			if(v[0] > 0) : 
				ta = 2*np.pi - ta

	a = h_mag*h_mag/(mu*(1-e_mag*e_mag))


	return (a, e_mag, i, W, w, ta)

def moon_reached(t, r, T=None, eps=None):

	return np.linalg.norm(r[:3]) - cst.d_M

def apside_pass(t, r, T=None, eps=None):
	""" Determines if the S/C passes either at the perigee and/or apogee. Detected by the x-velocity component change of 
		sign ((+) -> (-) for apogee, (-) -> (+) for perigee)"""
	v_x = r[3]

	return v_x
