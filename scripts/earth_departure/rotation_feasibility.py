import sys
import pykep as pk
import numpy as np 
import matplotlib.pyplot as plt

from scripts.earth_departure import constants as cst
from scripts.earth_departure.utils import R2, P_HRV2GEO, cart2sph, P_RVH2GEO
from scripts.utils import load_bodies, load_kernels

"""
	Given an inbound and an outbound velocity, this script determines if a single LGA is sufficient 
	or if the S/C must enter in resonance w/ the Moon and perform a 2nd LGA.

	The inbound velocity is computed with the ``apogee_raising`` script that provides the S/C velocity
	at Moon encounter in the Earth centered inertial frame. 
	The outbound velocity is computed with the ``Earth2NEA`` script that provides the S/C velocity relative
	to the Moon at Moon departure in the ECLJ2000 reference frame.

	Parameters:
	-----------
		v_in : [km] | [km/s]
			S/C velocity at Moon arrival in the Earth centered frame
		v_out : [km/s]
			S/C velocity relative to the Moon at departure in the ECLJ2000 frame
		tau : [MJD2000]
			Moon departure date
		r_m : [km]
			S/C - Moon minimal distance (relative to the Moon surface)

	Returns:
	--------
		feasible: bool
			True if one LGA is sufficient, False in the other case

"""


def rotation_feasibility(v_in, v_out, tau, r_m, gamma, print_=True):

	# 1 - Loading the SPICE kernels and the planet objects
	# ----------------------------------------------------
	load_kernels.load()

	t0 = pk.epoch(tau, julian_date_type='mjd2000')

	earth = load_bodies.planet(name='EARTH')
	moon  = load_bodies.planet(name='MOON')


	# 2 - Computation of the passage matrix from the ECLIPJ2000 frame to the HRV 
	# --------------------------------------------------------------------------

	r_E_ECLP, v_E_ECLP = earth.eph(t0)
	r_E_ECLP, v_E_ECLP = np.array(r_E_ECLP), np.array(v_E_ECLP)

	r_M_ECLP, v_M_ECLP = moon.eph(t0)
	r_M_ECLP, v_M_ECLP = np.array(r_M_ECLP), np.array(v_M_ECLP)

	# Computation of the Moon position, velocity and angular momentum unitary vectors in relation to the Earth
	r_M_rel = (r_M_ECLP - r_E_ECLP) / np.linalg.norm(r_M_ECLP - r_E_ECLP)
	v_M_rel = (v_M_ECLP - v_E_ECLP) / np.linalg.norm(v_M_ECLP - v_E_ECLP)
	h_M_rel = np.cross(r_M_ECLP, v_M_ECLP) / np.linalg.norm(np.cross(r_M_ECLP, v_M_ECLP))

	# Computation of the passage matrix from the ECLPJ2000 frame to HRV frame
	P_ECL2HRV = np.array([[h_M_rel[0], r_M_rel[0], v_M_rel[0]], 
			              [h_M_rel[1], r_M_rel[1], v_M_rel[1]],
			              [h_M_rel[2], r_M_rel[2], v_M_rel[2]]])

	# 3 - Computation of the inbound excess velocity in the Moon's HRV rotating frame
	# -------------------------------------------------------------------------------
	# Moon's velocity in the HRV basis
	v_M = np.array([0, 0, cst.V_M])

	# S/C velocity in the HRV basis
	v_in = P_HRV2GEO(gamma).dot(v_in)

	# Excess velocity
	v_inf_m = v_in - v_M

	# Computation of the longitude and co-latitude of the excess velocity vector before LGA
	v_inf_mag, phi_m, theta_m = cart2sph(v_inf_m)


	# 4 - Convertion of the escape excess velocity (after LGA) in the HRV frame
	# -------------------------------------------------------------------------
	v_inf_p = np.linalg.inv(P_ECL2HRV).dot(v_out)

	# Computation of the longitude and co-latitude of the excess velocity vector after LGA
	_, phi_p, theta_p = cart2sph(v_inf_p)

	# 5 - Computation of the angle between the velocities at infinity before and after the LGA and comparaison with the maximal rotation [rad]
	# ----------------------------------------------------------------------------------------------------------------------------------------
	# Angle between the velocities at infinity before and after the LGA [rad]
	delta = np.arccos( np.dot(v_inf_m, v_inf_p) / np.linalg.norm(v_inf_m) / np.linalg.norm(v_inf_p) )

	# Maximum rotation angle [rad]
	delta_m = 2 * np.arcsin( cst.mu_M / (cst.R_M+r_m) / (v_inf_mag**2 + cst.mu_M / (cst.R_M+r_m)) )

	if print_ == True:
		print("Delta     : {}°".format(delta   * 180 / np.pi))
		print("Delta max : {}°".format(delta_m * 180 / np.pi))

	feasibility = True if abs(delta) <= abs(delta_m) else False
	
	return feasibility, phi_m, theta_m, phi_p, theta_p


