
import matplotlib.pyplot as plt
import numpy as np 
import sys

from scripts.earth_departure.moon_moon_leg import moon_moon_leg, plot_env
from scripts.earth_departure import constants as cst
from scripts.earth_departure.utils import R2, cart2sph, sph2cart

"""
	In this script, we deal with the case where 1 LGA isn't sufficient and where our aim is to enter in resonance
	with the Moon to attempt a 2nd LGA.

	We define a p:q resonance ratio and try to find a suitable couple (phi_p, theta_p) allowing our S/C to catch
	the Moon a second time after p revolutions.

	Parameters:
	-----------

		v_inf_mag : [km/s]
			Velocity at infinity before and after the LGA 
		phi_m, theta_m : [rad]
			Azimuthal and polar angles of the velocity at infinity vector before the LGA
		r_p : [km]
			Minimal distance between the S/C and the Moon surface during the encounter
		p : [-]
			Number of revolution the S/C will accomplish before 2nd Moon encouter
		q : [-]
			Number of revolution the Moon will accomplish before 2nd S/C encouter

"""

def resonant_trajectories(v_inf_mag, phi_m, theta_m, phi_p, theta_p, r_m, gamma, p=1, q=1):

	# 1 - Computation of the polar angle of the S/C excess velocity after LGA to enter in p:q resonance with the Moon [rad]
	# ---------------------------------------------------------------------------------------------------------------------

	# S/C velocity after LGA to enter in a p:q resonance with the Moon [km/s]
	v = np.sqrt( 2*cst.mu_E/cst.d_M - (2*np.pi * cst.mu_E / (cst.T_M * p/q))**(2/3) )

	# Polar angle of the S/C velocity at infinity after LGA [rad]
	theta_p = np.arccos( (v**2 - cst.V_M**2 - v_inf_mag**2) / (2 * cst.V_M * v_inf_mag) )


	# 4 - Computation of the admissible longitude angles of the S/C velocity at infinity after LGA to enter in p:q resonance with the Moon [rad]
	# ------------------------------------------------------------------------------------------------------------------------------------------

	# Computation of the maximum rotation [rad]
	delta_max = 2 * np.arcsin( cst.mu_M/(cst.R_M+r_m) / (v_inf_mag**2 + cst.mu_M/(cst.R_M+r_m)) )

	# Possible longitude angles [rad]
	phi_p_arr = np.linspace(-np.pi, np.pi, 100)

	# Admissible longitude angles [rad]
	phi_p_adm = np.array([])

	def admissible_longitude(phi_p):
		return (np.cos(phi_m)*np.sin(theta_m)*np.sin(theta_p))*np.cos(phi_p) + \
			   (np.sin(phi_m)*np.sin(theta_m)*np.sin(theta_p))*np.sin(phi_p) + \
				np.cos(theta_m)*np.cos(theta_p) - np.cos(delta_max)

	for phi_p in phi_p_arr:
		if admissible_longitude(phi_p) >= 0:
			phi_p_adm = np.append(phi_p_adm, phi_p)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(phi_p_arr, np.array([ (np.cos(phi_m)*np.sin(theta_m)*np.sin(theta_p))*np.cos(phi_p) + \
						      	  (np.sin(phi_m)*np.sin(theta_m)*np.sin(theta_p))*np.sin(phi_p) + \
						    	  np.cos(theta_m)*np.cos(theta_p) - np.cos(delta_max) for phi_p in phi_p_arr]))

	plt.title("Feasible longitudes")
	plt.grid()
	plt.show()

	if len(phi_p_adm) == 0:
		print("No admissible solution with ({}:{}) resonance.".format(p, q), flush=True)
		sys.exit()

	r_fs = np.ndarray(shape=(0, 8))

	# 6 - Propagation of the Keplerian trajectory after the LGA
	# ---------------------------------------------------------
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	for phi_p in phi_p_adm:
		r_f = moon_moon_leg(v_inf_mag, phi_p, theta_p, gamma, p, q, ax)

		r_fs = np.append(r_fs, np.concatenate(([phi_p], [theta_p], r_f)))
	
	r_fs = r_fs.reshape(int(len(r_fs)/8), 8)

	plot_env(ax, gamma, p, q)

	ax.set_xlabel('X [km]')
	ax.set_xlabel('Y [km]')
	ax.set_xlabel('Z [km]')

	plt.legend()
	plt.show()

	return r_fs

	