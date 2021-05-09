import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.lga.cr3bp import CR3BP
from scripts.lga.coc import EM2SE, SE2EM
from scripts.lga.utils import initial_states, plot_env_2d, moon_orbit_intercept, earth_collision, moon_rot_matrix, secant_mtd_f
from scripts.lga.escape import second_LGA

# 1 - Construction of the figure object
# -------------------------------------
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)


# 2 - Definition of the Sun-Earth CR3BP
# -------------------------------------
cr3bp = CR3BP(mu=3.036e-6, L=1.496e8, V=29.784, T=3.147e7)

# 3 - Definition of the free parameters
# -------------------------------------
v_inf = 0.3 # Velocity at infinity [km/s]
# alpha = np.array([120, 122]) * np.pi / 180
alpha = np.linspace(0, 180, 180) * np.pi / 180 # Angle between v_inf and v_M [rad]
theta_s0 = 80 * np.pi / 180 # Angle between the earth-sun axis and the earth-moon axis [rad]

# 4 - Definition of other variables
# ---------------------------------
beta_rcd = 0
alpha_double_lga = np.array([])

# 5 - Main process
# ----------------
for k, alpha_ in enumerate(alpha):

	# - * - * - * - * - * - * - * - * - * - * - * - 
	print("Alpha ... {}°".format(alpha_*180/np.pi))
	# - * - * - * - * - * - * - * - * - * - * - * - 

	# 5.1 - Computation of the initial state
	# ------------------------------------
	r0 = initial_states(cr3bp, v_inf, alpha_, theta_s0)

	# 5.2 - Integration of the CR3BP equations in the Sun-Earth synodic frame
	# ---------------------------------------------------------------------
	t_span = [0, 2]
	t_eval = np.linspace(t_span[0], t_span[1], 1000)
	moon_orbit_intercept.terminal = True
	moon_orbit_intercept.direction = 1
	earth_collision.terminal = True
	sol = solve_ivp(fun=cr3bp.states_derivatives, t_span=t_span, y0=r0, method='RK45', events=(moon_orbit_intercept, earth_collision), \
		rtol=1e-13, atol=1e-12)

	t = sol.t
	r = sol.y

	# for i in range(len(r[0])):
	# 	r[:, i] = SE2EM(cr3bp, r[:, i], theta_s0)

	# ax.plot(r[0], r[1], color='blue')
	# plot_env(ax, cr3bp, theta_s0)

	# 5.3 Check if a Lunar encounter has been found
	# ---------------------------------------------
	r_Earth = np.array([1-cr3bp.mu, 0, 0])
	R_M = 384402 / cr3bp.L
	V_M = 1.0220 / cr3bp.V

	if abs(np.linalg.norm(r[0:3, -1] - r_Earth) - R_M) < 1e-10:

		# 5.3.1 - Conversion of the states in the Earth-Moon frame
		# ------------------------------------------------------
		for i in range(len(r[0])):
			r[:, i] = SE2EM(cr3bp, r[:, i], theta_s0)

		# 5.3.2 - Computation of the Moon position at t = tf
		# --------------------------------------------------
		tf = t[-1]
		n_per = tf*cr3bp.T / 2551392
		
		rotation_angle = 2 * np.pi * n_per

		r_M_i = np.array([R_M, 0, 0, 0, V_M, 0])
		r_M_f = moon_rot_matrix(rotation_angle).dot(r_M_i)

		# 5.3.3 - Computation of the angle beta between the S/C and the Moon at t=tf
		# --------------------------------------------------------------------------
		r_f_unt = r[0:3, -1] / np.linalg.norm(r[0:3, -1])
		r_M_unt = r_M_f[0:3] / np.linalg.norm(r_M_f[0:3])

		beta = np.sign(np.cross(r_M_unt, r_f_unt)[2]) * np.arccos(r_M_unt.dot(r_f_unt))

		# 5.3.4 - Check if beta changed of sign
		# -------------------------------------
		if k == 0:
			beta_rcd = beta

		else:
			if beta*beta_rcd < 0:

				print("\tResearch for a double LGA w/ alpha in [{}°, {}°]".format(alpha[k-1]*180/np.pi, alpha[k]*180/np.pi))

				alpha_0 = alpha[k-1]
				alpha_1 = alpha[k]

				beta_0 = secant_mtd_f(cr3bp, v_inf, alpha_0, theta_s0, t_span)
				beta_1 = secant_mtd_f(cr3bp, v_inf, alpha_1, theta_s0, t_span)

				alpha_n = alpha_0
				alpha_nn = alpha_1
				alpha_nnn = 1

				beta_n = beta_0
				beta_nn = beta_1
				beta_nnn = 1

				count = 0

				while abs(beta_nnn) > 1e-10 and count < 50:
					# To avoid division by 0 error
					denominator = (beta_nn - beta_n)
					if denominator == 0:
						# - * - * - * - * - * - * - * - * - * - *
						print("\tSecants method didn't converge.")
						# - * - * - * - * - * - * - * - * - * - *
						break

					alpha_nnn = alpha_nn - (alpha_nn - alpha_n) / denominator * beta_nn

					# Computation of the f(alpha_n) value
					beta_nnn = secant_mtd_f(cr3bp, v_inf, alpha_nnn, theta_s0, t_span)

					# Update the variables
					alpha_n = alpha_nn
					beta_n = beta_nn

					alpha_nn = alpha_nnn
					beta_nn = beta_nnn

					# Increment the counter
					count += 1

				if abs(beta_nnn) < 1e-10 and count < 50:
					alpha_double_lga = np.append(alpha_double_lga, alpha_nnn)
					# - * - * - * - * - * - * - * - * - * - *
					print("\tFound a double LGA solution : {}° !".format(alpha_nnn*180/np.pi))
					# - * - * - * - * - * - * - * - * - * - *
				else:
					# - * - * - * - * - * - * - * - * - * - *
					print("\tSecants method didn't converge.")
					# - * - * - * - * - * - * - * - * - * - *

			beta_rcd = beta


		# 5.3.5 - Plot of the computed trajectory in the Earth-Moon frame
		# -------------------------------------------------------------
		ax.plot(r[0], r[1], color='blue')
		ax.plot([r[0, -1]], [r[1, -1]], 'o', color='m', markersize=3)

# 6 - Plot of the environment and all the trajectories
# ----------------------------------------------------
plot_env_2d(ax, cr3bp, theta_s0, "Trajectories intercepting Moon's orbit")

# 7 - Post processing of the double LGA trajectories
# --------------------------------------------------

fig_final = plt.figure(figsize=(7, 7))
ax_final = fig_final.add_subplot(111)

for k, alpha_ in enumerate(alpha_double_lga):
	# - * - * - * - * - * - * - * - * - * - * - * - 
	print("Alpha ... {}°".format(alpha_*180/np.pi))
	# - * - * - * - * - * - * - * - * - * - * - * - 

	# Computation of the initial state
	r0 = initial_states(cr3bp, v_inf, alpha_, theta_s0)

	# Integration of the CR3BP equations in the Sun-Earth synodic frame
	t_span = [0, 2]
	t_eval = np.linspace(t_span[0], t_span[1], 1000)
	sol = solve_ivp(fun=cr3bp.states_derivatives, t_span=t_span, y0=r0, method='RK45', events=(moon_orbit_intercept, earth_collision), \
		rtol=1e-13, atol=1e-12)

	t = sol.t
	r = sol.y

	# Conversion of the states in the Earth-Moon frame
	for i in range(len(r[0])):
		r[:, i] = SE2EM(cr3bp, r[:, i], theta_s0)

	ax_final.plot(r[0], r[1], color='blue')
	ax_final.plot([r[0, -1]], [r[1, -1]], 'o', color='m', markersize=3)

plot_env_2d(ax_final, cr3bp, theta_s0, "Feasible double LGA")


# # Call the escape study function
# for alpha_ in alpha_double_lga:
# 	second_LGA(cr3bp, v_inf, alpha_, theta_s0)
