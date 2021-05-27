import sys
import math as mt
import numpy as np 
import pykep as pk

import matplotlib.pyplot as plt

from scripts.utils import load_bodies, load_kernels

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

def P_ECI_HRV():
	""" Transition matrix from the ECI frame to the HRV one """
	return np.array([[0, 1, 0],
					 [0, 0, 1], 
					 [1, 0, 0]])

def P_HRV_ECI():
	""" Transition matrix from the ECI frame to the HRV one """
	return np.array([[0, 0, 1],
					 [1, 0, 0], 
					 [0, 1, 0]])

def P_ECLJ2000_ECI(tau):
	""" Transition matrix from the ECLJ2000 frame and the ECI one with axis following :
			i : Completing the right-hand rule    !!! The Moon is not on the x-axis at t=0 !!!
			j : Moon unitary velocity vector w.r.t the Earth [-]
			k : Moon's angular momentum w.r.t the Earth [-] 		"""

	load_kernels.load()

	t0 = pk.epoch(tau, julian_date_type='mjd2000')

	earth = load_bodies.planet(name='EARTH')
	moon  = load_bodies.planet(name='MOON')

	# 2 - Computation of the passage matrix from the ECLIPJ2000 frame to the ECI 
	# --------------------------------------------------------------------------
	r_E_ECLP, v_E_ECLP = earth.eph(t0)
	r_E_ECLP, v_E_ECLP = np.array(r_E_ECLP) / 1000, np.array(v_E_ECLP) / 1000

	r_M_ECLP, v_M_ECLP = moon.eph(t0)
	r_M_ECLP, v_M_ECLP = np.array(r_M_ECLP) / 1000, np.array(v_M_ECLP) / 1000

	# Unitary vector of the ECI frame
	j = (v_M_ECLP - v_E_ECLP) / np.linalg.norm(v_M_ECLP - v_E_ECLP)
	k = np.cross(r_M_ECLP - r_E_ECLP, v_M_ECLP - v_E_ECLP) / np.linalg.norm(np.cross(r_M_ECLP - r_E_ECLP, v_M_ECLP - v_E_ECLP))
	i = np.cross(j, k) / np.linalg.norm(np.cross(j, k))

	return np.array([[i[0], j[0], k[0]],
					 [i[1], j[1], k[1]],
					 [i[2], j[2], k[2]]])

def ECLJ2000_ECI(tau, r_ECLJ2000):
	""" Conversion of S/C states from the ECLJ2000 frame to the ECI one """

	# 1 - Loading the SPICE kernels and the planet objects
	# ----------------------------------------------------
	load_kernels.load()

	t0 = pk.epoch(tau, julian_date_type='mjd2000')

	earth = load_bodies.planet(name='EARTH')
	moon  = load_bodies.planet(name='MOON')

	# 2 - Computation of the passage matrix from the ECLIPJ2000 frame to the ECI 
	# --------------------------------------------------------------------------
	r_E_ECLP, v_E_ECLP = earth.eph(t0)
	r_E_ECLP, v_E_ECLP = np.array(r_E_ECLP) / 1000, np.array(v_E_ECLP) / 1000

	r_M_ECLP, v_M_ECLP = moon.eph(t0)
	r_M_ECLP, v_M_ECLP = np.array(r_M_ECLP) / 1000, np.array(v_M_ECLP) / 1000

	# Unitary vector of the ECI frame
	i = (r_M_ECLP - r_E_ECLP) / np.linalg.norm(r_M_ECLP - r_E_ECLP)
	k = np.cross(r_M_ECLP - r_E_ECLP, v_M_ECLP - v_E_ECLP) / np.linalg.norm(np.cross(r_M_ECLP - r_E_ECLP, v_M_ECLP - v_E_ECLP))
	j = np.cross(k, i) / np.linalg.norm(np.cross(k, i))

	# Passage matrix from the ECLJ2000 frame to the ECI one
	P = np.array([[i[0], j[0], k[0]],
				  [i[1], j[1], k[1]],
				  [i[2], j[2], k[2]]])

	# Computation of the S/C position and velocity relatively to the Earth in the ECLJ2000
	r = r_ECLJ2000 - np.concatenate((r_E_ECLP, v_E_ECLP))

	# Computation of the S/C position and velocity relatively to the Earth in the ECI
	r[:3], r[3:] = np.linalg.inv(P).dot(r[:3]), np.linalg.inv(P).dot(r[3:])

	return r

