import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import solve_ivp

from scripts.lga.resonance.utils_2 import kepler, R1

def moon_moon_leg(v_inf_mag, phi, theta, p, q, ax):
	""" Propagate the Keplerian trajectory of a S/C after a LGA, targeting the Moon
		for a second time """

	# 1 - Definition of Moon's and Earth's characteristics
	# ----------------------------------------------------

	# Earth's gravitational parameter [km^3/s^2]
	mu_E = 398600.4418

	# Earth-Moon mean distance [km]
	d_M = 385000

	# Moon's gravitational parameter [km^3/s^2]
	mu_M = 4902.7779

	# Moon's radius [km]
	R_M = 1737.4

	# Moon's orbital velocity [km/s]
	V_M = 1.01750961806616

	# Moon's orbit period [s]
	T_M = 2377399


	# 2 - Definition of the vectors in the Moon centered frame
	# --------------------------------------------------------

	# Moon's velocity [km/s]
	v_M = np.array([0, 0, V_M])

	# S/C velocity at infinity after the LGA [km/s]
	v_inf = v_inf_mag * np.array([ np.cos(phi) * np.sin(theta),
								   np.sin(phi) * np.sin(theta),
								   np.cos(theta)              ])

	# S/C velocity relative to the Moon [km/s]
	v = v_M + v_inf

	# 3 - Basis changement Moon rotating (1) -> Moon rotating (2) -> Earth centered
	# -----------------------------------------------------------------------------
	v   = R1().dot(v)
	v_M = R1().dot(v_M)


	# 4 - Construction of the initial states of the Moon and the S/C
	# --------------------------------------------------------------
	r_0 = np.array([d_M, 0, 0, v[0], v[1], v[2]])
	r_M_0 = np.array([d_M, 0, 0, v_M[0], v_M[1], v_M[2]])


	# 5 - Propagation of the Keplerian equations
	# ------------------------------------------
	t_span = [0, max(p, q)*T_M]
	t_eval = np.linspace(t_span[0], t_span[-1], 1000)

	sol_S = solve_ivp(fun=kepler, t_span=t_span, y0=r_0, t_eval=t_eval, rtol=1e-13, atol=1e-14)
	r_S = sol_S.y

	ax.plot(r_S[0], r_S[1], r_S[2], '-', linewidth=1, color='blue')
	ax.plot([r_S[0, -1]], [r_S[1, -1]], [r_S[2, -1]], 'o', markersize=2, color='green')


def plot_env(ax, p, q):

	# 1 - Definition of Moon's and Earth's characteristics
	# ----------------------------------------------------

	# Earth's gravitational parameter [km^3/s^2]
	mu_E = 398600.4418

	# Earth-Moon mean distance [km]
	d_M = 385000

	# Moon's gravitational parameter [km^3/s^2]
	mu_M = 4902.7779

	# Moon's radius [km]
	R_M = 1737.4

	# Moon's orbital velocity [km/s]
	V_M = 1.01750961806616

	# Moon's orbit period [s]
	T_M = 2377399


	# 2 - Moon's initial states
	# -------------------------
	r_M_0 = np.array([d_M, 0, 0, 0, V_M, 0])

	# 3 - Propagation of the Keplerian equations
	# ------------------------------------------
	t_span = [0, max(p, q)*T_M]
	t_eval = np.linspace(t_span[0], t_span[-1], 1000)

	sol_M = solve_ivp(fun=kepler, t_span=t_span, y0=r_M_0, t_eval=t_eval, rtol=1e-13, atol=1e-14)
	r_M = sol_M.y

	ax.plot([0], [0], [0], 'o', markersize=8, color='black')
	ax.plot(r_M[0], r_M[1], r_M[2], '-', linewidth=1, color='black')

	ax.plot([r_M[0, -1]], [r_M[1, -1]], [r_M[2, -1]], 'o', markersize=2, color='red')

