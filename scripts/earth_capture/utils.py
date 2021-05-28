import numpy as np 
import math as mt 

import matplotlib.pyplot as plt

from scripts.earth_capture import constants as cst


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


def C3(r):
	""" Computation of a S/C characteristic energy in a keplerian approximation.	"""
	
	# S/C position and velocity w.r.t the Earth [km] | [km/s]
	d, v = np.linalg.norm(r[:3]), np.linalg.norm(r[3:])

	return 2 * (v**2/2 - cst.mu_E/d)



def kep2cart(a, e, i, W, w, ta, mu) : 

	h = np.sqrt(mu*a*(1-e**2))

	r_orbital_plan = (h*h/mu)*(1/(1+e*np.cos(ta)))*np.array([np.cos(ta), np.sin(ta), 0])
	v_orbital_plan = mu/h*np.array([-np.sin(ta), e+np.cos(ta), 0])


	R1 = np.array([ [np.cos(w), -np.sin(w), 0.],
					[np.sin(w),  np.cos(w), 0.],
					[			 0.,			   0., 1.] ])

	R2 = np.array([ [1.,			   0.,			   0.],
					[0.,	  np.cos(i),	 -np.sin(i)],
					[0.,	  np.sin(i),	  np.cos(i)] ])

	R3 = np.array([ [np.cos(W), -np.sin(W),	 0.],
					[np.sin(W),  np.cos(W),	 0.],
					[			 0.,			   0.,	 1.] ])

	r = R3.dot(R2.dot(R1.dot(r_orbital_plan)))
	v = R3.dot(R2.dot(R1.dot(v_orbital_plan)))
	
	return np.concatenate((r, v))


def cart2kep(r, v, mu) : 

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

	return a, e_mag, i, W, w, ta


def plot_env_2D(ax):
	""" Plot the Earth, and the Moon's orbit on a pre-defined 2D-figure """

	ax.plot([0], [0], 'o', markersize=8, color='black', label='Earth')
	ax.plot([cst.d_M * np.cos(t_) for t_ in np.linspace(0, 2*np.pi, 1000)], [cst.d_M * np.sin(t_) for t_ in np.linspace(0, 2*np.pi, 1000)], '-', \
		linewidth=1, color='black', label='Moon trajectory')

def plot_env_3D(ax):
	""" Plot the Earth, and the Moon's orbit on a pre-defined 2D-figure """

	ax.plot([0], [0], [0], 'o', markersize=8, color='black', label='Earth')
	ax.plot([cst.d_M * np.cos(t_) for t_ in np.linspace(0, 2*np.pi, 1000)], [cst.d_M * np.sin(t_) for t_ in np.linspace(0, 2*np.pi, 1000)], \
		np.zeros(1000), '-', linewidth=1, color='black', label='Moon trajectory')


def angle_w_Ox(r):
	""" Returns the (oriented) angle between the Ox axis and the ``r`` vector """
	return np.sign(np.cross(np.array([1, 0, 0]), r)[2]) * np.arccos( np.dot(np.array([1, 0, 0]), r) / np.linalg.norm(r) )
	