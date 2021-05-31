import sys
import numpy as np 

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.earth_departure import constants as cst
from scripts.earth_capture.coc import P_ECLJ2000_ECI, P_ECI_HRV, cart2sph, sph2cart
from scripts.earth_capture.utils import kepler, cart2kep, plot_env_2D, plot_env_3D, C3, cart2kep
from scripts.utils import load_bodies, load_kernels
			

def first_lga(r, r_m, p, q):
	""" Computes the Characteristic Energy that can be reached given the S/C state before the LGA and the 
		minimal fly-by radius.

		Parameters
		----------
		r : array
			S/C state in the ECI frame before the LGA [km] | [km/s]
		r_m : float
			Fly-by minimal radius (relatively to the Moon's surface) [km]
		p : float
			Number of revolution the S/C will accomplish before 2nd Moon encouter [-]
		q : float
			Number of revolution the Moon will accomplish before 2nd S/C encouter [-]

		Returns
		-------
		r_fs: ndarray
			Matrix containing the final S/C position in the ECI frame and the post-LGA spherical angles

	"""

	# Computation of the S/C velocity in the HRV frame
	P_HRV_ECI = np.linalg.inv(P_ECI_HRV())
	v_HRV = P_HRV_ECI.dot(r[3:])

	# Moon's velocity in the HRV
	v_M_HRV = np.array([0, 0, cst.V_M])

	v_inf, phi_m, theta_m = cart2sph(v_HRV - v_M_HRV)

	# S/C distance to the Earth [km]
	d = np.linalg.norm(r[:3])

	# 1 - Computation of the polar angle of the S/C excess velocity after LGA to enter in p:q resonance with the Moon [rad]
	# ---------------------------------------------------------------------------------------------------------------------
	# S/C velocity after LGA to enter in a p:q resonance with the Moon [km/s]
	v = np.sqrt( 2*cst.mu_E/d - (2*np.pi * cst.mu_E / (cst.T_M * p/q))**(2/3) )

	# Polar angle of the S/C velocity at infinity after LGA [rad]
	theta_p = np.arccos( (v**2 - cst.V_M**2 - v_inf**2) / (2 * cst.V_M * v_inf) )


	# 2 - Computation of the admissible longitude angles of the S/C velocity at infinity after LGA to enter in p:q resonance with the Moon [rad]
	# ------------------------------------------------------------------------------------------------------------------------------------------
	# Computation of the maximum rotation [rad]
	delta_max = 2 * np.arcsin( cst.mu_M/(cst.R_M+r_m) / (v_inf**2 + cst.mu_M/(cst.R_M+r_m)) )

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


	r_fs = np.ndarray(shape=(0, 8))

	t_span = [0, p/q * cst.T_M]
	t_eval = np.linspace(t_span[0], t_span[-1], 10000)

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	for k, phi_p in enumerate(phi_p_adm):

		# Computation of the post-LGA S/C velocity in the ECI frame
		v_HRV_p = sph2cart([v_inf, phi_p, theta_p]) + v_M_HRV
		v_ECI_p = P_ECI_HRV().dot(v_HRV_p)
		r0 = np.concatenate((r[:3], v_ECI_p))

		solution = solve_ivp(fun=kepler, t_span=t_span, t_eval=t_eval, y0=r0, rtol=1e-12, atol=1e-12)
		r_f = solution.y[:, -1]

		r_fs = np.append(r_fs, np.concatenate(([phi_p], [theta_p], r_f)))

		ax.plot(solution.y[0], solution.y[1], solution.y[2], '-', color='blue', linewidth=1)

	plot_env_3D(ax)
	plt.show()

	r_fs = r_fs.reshape(int(len(r_fs)/8), 8)

	return r_fs, solution.t

