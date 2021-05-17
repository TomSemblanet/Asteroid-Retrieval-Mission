import sys
import numpy as np 
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.rotation_feasibility import rotation_feasibility
from scripts.earth_departure.resonance_study import resonant_trajectories
from scripts.earth_departure.utils import kepler, R2, cart2sph, sph2cart, P_GEO2HRV, angle_w_Ox, plot_env_3D
from scripts.earth_departure import constants as cst

def one_lga(v_inf, phi_m, theta_m, phi_p, theta_p, gamma, r_ar, t_ar):

	print("Solution found with one lunar gravity assist\n")
	print("({}°, {}°) -> ({}°, {}°)".format(phi_m*180/np.pi, theta_m*180/np.pi, phi_p*180/np.pi, theta_p*180/np.pi))


	# Propagation of the outter leg
	v_out = P_GEO2HRV(gamma).dot(sph2cart([v_inf, phi_p, theta_p]))
	r0 = np.concatenate((r_ar[:3, -1], v_out))

	t_span = np.array([0, 10*86400])
	t_eval = np.linspace(t_span[0], t_span[-1], 10000)

	sol = solve_ivp(fun=kepler, t_span=t_span, t_eval=t_eval, y0=r0, rtol=1e-12, atol=1e-12)


	# Rotation of the frame so that at t=tf, the (Ox) ax is following the Earth-Moon vector
	for k in range(len(sol.y[0])):
		sol.y[:3, k] = R2(-gamma).dot(sol.y[:3, k])
		sol.y[3:, k] = R2(-gamma).dot(sol.y[3:, k])

	for k in range(len(r_ar[0])):
		r_ar[:3, k] = R2(-gamma).dot(r_ar[:3, k])
		r_ar[3:, k] = R2(-gamma).dot(r_ar[3:, k])


	# Plot of the solution
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	plot_env_3D(ax=ax)
	ax.plot(r_ar[0], r_ar[1], r_ar[2], '-', linewidth=1, color='blue')
	ax.plot(sol.y[0], sol.y[1], sol.y[2], '-', linewidth=1, color='blue')

	plt.legend()
	plt.show()

	trajectories = [r_ar, sol.y]
	time = [t_ar, sol.t]

	return trajectories, time