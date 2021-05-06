import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.lga.cr3bp import CR3BP
from scripts.lga.coc import EM2SE, SE2EM
from scripts.lga.utils import initial_states, plot_env_3d, moon_orbit_intercept, earth_collision, moon_rot_matrix

def EM2R(theta):
	""" Rotation matrix between the Earth-Moon frame and the rotating one """
	return np.array([[np.cos(theta), -np.sin(theta), 0],
				     [np.sin(theta),  np.cos(theta), 0],
				     [            0,              0, 1]])

def second_LGA(cr3bp, v_inf, alpha, theta_s0):
	""" Once a Moon - Moon transfer has been found, this function studies the 2nd LGA
		and the produced trajectory 

		Parameters
		----------
		cr3bp : <CR3BP object>
			CR3BP object, representing the S/C environment
		alpha : float 
			Initial angle between V_M and V_inf [rad]
		thetas_s0 : float
			Initial angle between the Moon and the Sun [rad]
		v_inf : float 
			Velocity at infinity at 1st Moon departure [-]
	"""

	fig = plt.figure(figsize=(7, 7))
	ax = fig.gca(projection='3d')

	# Some constants
	R_M = 384402 / cr3bp.L # Earth - Moon distance [-]
	V_M = 1.0220 / cr3bp.V # Moon velocity in the Earth-centered frame [-]
	mu_E = 398600.4418 # Gravitational parameter of the Earth [km^3/s^2]
	mu_M = 4902.7779 # Gravitational parameter of the Moon [km^3/s^2]
	rp = 1738.1 + 50 # LGA perigee [km]

	# 1 - Preparation of the parameters for the propagation
	# -----------------------------------------------------
	r0 = initial_states(cr3bp, v_inf, alpha, theta_s0)
	t_span = [0, 2]
	t_eval = np.linspace(t_span[0], t_span[1], 1000)

	# 2 - Propagation of the CR3BP equations until 2nd LGA point is reached
	# ---------------------------------------------------------------------
	sol = solve_ivp(fun=cr3bp.states_derivatives, t_span=t_span, y0=r0, method='RK45', events=(moon_orbit_intercept, earth_collision), \
		rtol=1e-13, atol=1e-12)
	r, t = sol.y, sol.t

	# 3 - Conversion of the states in the Earth-Moon frame
	# ----------------------------------------------------
	for i in range(len(r[0])):
		r[:, i] = SE2EM(cr3bp, r[:, i], theta_s0)

	# 4 - Computation of the Moon state at t=tf
	# -------------------------------------
	tf = t[-1]
	n_per = tf*cr3bp.T / 2551392
	theta = 2 * np.pi * n_per

	r_M_i = np.array([R_M, 0, 0, 0, V_M, 0])
	r_M_f = moon_rot_matrix(theta).dot(r_M_i)

	# Creation of the rotated frame
	R = EM2R(theta)
	i = R.dot(np.array([1, 0, 0]))
	j = R.dot(np.array([0, 1, 0]))
	k = R.dot(np.array([0, 0, 1]))

	# ax.plot([r_M_f[0], r_M_f[0]+i[0]/500], [r_M_f[1], r_M_f[1]+i[1]/500], ':', label='u')
	# ax.plot([r_M_f[0], r_M_f[0]+j[0]/500], [r_M_f[1], r_M_f[1]+j[1]/500], ':', label='v')

	# # Plot of the Moon velocity vector
	# ax.plot([r_M_f[0], r_M_f[0]+r_M_f[3]/100], [r_M_f[1], r_M_f[1]+r_M_f[4]/100], label='Moon velocity (t=tf)')
	# # Plot of the S/C velocity vector
	# ax.plot([r_M_f[0], r_M_f[0]+r[3, -1]/100], [r_M_f[1], r_M_f[1]+r[4, -1]/100], label='S/C Velocity (t=tf)')
	# # Plot of the S/C infinity velocity vector
	# ax.plot([r_M_f[0], r_M_f[0]+(r[3, -1]-r_M_f[3])/100], [r_M_f[1], r_M_f[1]+(r[4, -1]-r_M_f[4])/100], label='S/C V infinity')

	# 5 - Computation of the S/C velocity before the 2nd LGA
	# ------------------------------------------------------
	# S/C velocity in the Earth-Moon frame before LGA [-]
	V_m_SC = r[3:, -1]

	# S/C Velocity at infinity in the Earth-Moon frame before LGA [-]
	V_m_inf = V_m_SC - r_M_f[3:]
	V_inf_mag = np.linalg.norm(V_m_inf)

	# S/C Velocity at infinity in the rotated frame before LGA [-]
	# /!\ Si P est la matrice de passage de B à B' alors un vecteur X
	#     exprimé dans B s'exprimera dans B' via X' = P^(-1)*X /!\
	R_inv = EM2R(-theta)

	V_SC_rot = R_inv.dot(V_m_SC)
	V_M_rot  = np.array([0, V_M, 0])
	V_m_inf_rot = V_SC_rot - V_M_rot

	# Pump-angle before the LGA
	p_m = np.sign(np.cross(V_M_rot, V_m_inf_rot)[-1]) * np.arccos( np.dot(V_M_rot, V_m_inf_rot) / np.linalg.norm(V_m_inf_rot) / np.linalg.norm(V_M_rot)) 
	print("Pump angle : {}°".format(p_m * 180 / np.pi))

	# Maximum rotation angle (delta_max)
	delta_max = 2 * np.arcsin( mu_M / rp / ((V_inf_mag*cr3bp.V)**2 + mu_M / rp) )

	# Pump angle (+) after LGA interval
	p_p_range = np.linspace(p_m - delta_max, p_m + delta_max, 2)

	# Iteration on each post-LGA pump angles
	for p_p in p_p_range:
		# Computation of the `arg` value (cf. Master Thesis Alex Siniscalco)
		arg = (np.cos(delta_max) - np.cos(p_m)*np.cos(p_p)) / (np.sin(p_m)*np.sin(p_p))

		# Computation of sin(p-)*sin(p+) for case disjunction
		product = np.sin(p_m)*np.sin(p_p)

		# Computation of the maximal cranck angle
		if (product > 0 and arg < 1 and arg > -1):
			k_p_max = np.arccos(arg)

		elif (product > 0 and arg < -1):
			k_p_max = np.pi

		elif product < 0:
			k_p_max = np.pi

		else:
			continue

		k_p_range = np.linspace(0, k_p_max, 5)

		# Iteration on each post-LGA cranck angles
		for k_p in k_p_range:
			print("\np+ : {}°\tk+ : {}".format(p_p*180/np.pi, k_p*180/np.pi))

			# S/C Velocity at infinity relative to the Moon after LGA in the rotated frame [-]
			u_p_inf = V_inf_mag * np.sin(p_p) * np.cos(k_p)
			v_p_inf = V_inf_mag * np.cos(p_p)
			w_p_inf = V_inf_mag * np.sin(p_p) * np.sin(k_p)
			V_p_inf_rot = np.array([u_p_inf, v_p_inf, w_p_inf])

			# S/C Velocity after LGA in the rotated frame [-]
			V_p_SC_rot = V_p_inf_rot + np.array([0, V_M, 0])

			# S/C Velocity after LGA in the Earth-Moon frame [-]
			V_p_SC = R.dot(V_p_SC_rot)

			# Computation of the Energy
			E = (np.linalg.norm(V_p_SC*cr3bp.V)**2)/2 - mu_E/(R_M*cr3bp.L)
			print("C3 : {} km^2/s^2".format(2*E))

			# Computation of the S/C state in the Earth-Moon frame after LGA
			r0_ = np.concatenate((r_M_f[:3], V_p_SC))

			# Conversion in the Sun-Earth frame
			r0_ = EM2SE(cr3bp, r0_, theta_s0)

			# Propagation
			t_span_ = [0, 2]
			sol_ = sol = solve_ivp(fun=cr3bp.states_derivatives, t_span=t_span_, y0=r0_, method='RK45', events=(earth_collision), \
				rtol=1e-13, atol=1e-12)
			r_, t_ = sol.y, sol.t

			# Conversion in the Earth-Moon frame
			for i in range(len(r_[0])):
				r_[:, i] = SE2EM(cr3bp, r_[:, i], theta_s0)

			if t_[-1] == t_span_[-1]:
				ax.plot(r_[0], r_[1], r_[2], color='blue')


	ax.plot(r[0], r[1], r[2], color='blue')
	ax.plot([r[0, -1]], [r[1, -1]], [r[2, -1]], 'o', color='magenta', markersize=3)

	plot_env_3d(ax, cr3bp, theta_s0, "Feasible double LGA")



