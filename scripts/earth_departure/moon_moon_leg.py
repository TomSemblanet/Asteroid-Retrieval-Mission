import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import solve_ivp

from scripts.earth_departure.utils import kepler, P_GEO2HRV, P_GEO2RVH, sph2cart
from scripts.earth_departure import constants as cst

def moon_moon_leg(v_inf_mag, phi, theta, gamma, p, q, ax):
	""" Propagate the Keplerian trajectory of a S/C after a LGA, targeting the Moon
		for a second time """

	# 2 - Definition of the vectors in the HRV frame
	# ----------------------------------------------

	# Moon's velocity in the HRV frame [km/s]
	v_M = np.array([0, 0, cst.V_M])

	# S/C velocity at infinity after the LGA in the HRV frame [km/s]
	v_inf = sph2cart([v_inf_mag, phi, theta])

	# S/C velocity relative to the Earth in the HRV frame [km/s]
	v = v_M + v_inf

	# 3 - Basis changement from HRV to Earth inertial frame
	# -----------------------------------------------------
	v = P_GEO2HRV(gamma).dot(v)


	# 4 - Construction of the initial states of the S/C in the Earth inertial frame
	# -----------------------------------------------------------------------------
	r_0 = np.array([cst.d_M, 0, 0, v[0], v[1], v[2]])
	r_0[:3] = P_GEO2RVH(gamma).dot(r_0[:3])


	# 5 - Propagation of the Keplerian equations
	# ------------------------------------------
	t_span = [0, max(p, q)*cst.T_M]
	t_eval = np.linspace(t_span[0], t_span[-1], 1000)

	sol_S = solve_ivp(fun=kepler, t_span=t_span, y0=r_0, t_eval=t_eval, rtol=1e-13, atol=1e-14)
	r_S = sol_S.y

	ax.plot(r_S[0], r_S[1], r_S[2], '-', linewidth=1, color='blue')
	ax.plot([r_S[0, -1]], [r_S[1, -1]], [r_S[2, -1]], 'o', markersize=2, color='green')

	return r_S[:, -1]


def plot_env(ax, gamma, p, q):

	# 2 - Moon's initial states
	# -------------------------
	r_M_0 = np.array([cst.d_M, 0, 0, 0, cst.V_M, 0])
	r_M_0[:3], r_M_0[3:] = P_GEO2RVH(gamma).dot(r_M_0[:3]), P_GEO2RVH(gamma).dot(r_M_0[3:])

	# 3 - Propagation of the Keplerian equations
	# ------------------------------------------
	t_span = [0, max(p, q)*cst.T_M]
	t_eval = np.linspace(t_span[0], t_span[-1], 1000)

	sol_M = solve_ivp(fun=kepler, t_span=t_span, y0=r_M_0, t_eval=t_eval, rtol=1e-13, atol=1e-14)
	r_M = sol_M.y

	ax.plot([0], [0], [0], 'o', markersize=8, color='black')
	ax.plot(r_M[0], r_M[1], r_M[2], '-', linewidth=1, color='black')

	ax.plot([r_M[0, -1]], [r_M[1, -1]], [r_M[2, -1]], 'o', markersize=2, color='red')

