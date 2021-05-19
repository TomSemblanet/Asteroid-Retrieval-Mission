import sys
import numpy as np 
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.rotation_feasibility import rotation_feasibility
from scripts.earth_departure.resonance_study import resonant_trajectories
from scripts.earth_departure.utils import kepler, kepler_thrust, R2, cart2sph, sph2cart, P_GEO2HRV, angle_w_Ox, plot_env_3D
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
	sigma = 2 * np.pi * t_ar[-1] / cst.T_M

	Rot = R2(sigma-gamma)

	for k in range(len(t_ar)):
		r_ar[:3, k] = Rot.dot(r_ar[:3, k])
		r_ar[3:, k] = Rot.dot(r_ar[3:, k])

	for k in range(len(sol.t)):
		sol.y[:3, k] = Rot.dot(sol.y[:3, k])
		sol.y[3:, k] = Rot.dot(sol.y[3:, k])

	trajectories = [r_ar, sol.y]
	times = [t_ar, sol.t]

	return trajectories, times


def two_lga(v_inf, phi_1_m, theta_1_m, phi_1_p, theta_1_p, phi_2_m, theta_2_m, phi_2_p, theta_2_p, p, q, gamma, r_ar, t_ar):

	# Computation of the states in cartesian coordinates
	r_1_m = sph2cart([v_inf, phi_1_m, theta_1_m])
	r_1_p = sph2cart([v_inf, phi_1_p, theta_1_p])
	r_2_m = sph2cart([v_inf, phi_2_m, theta_2_m])
	r_2_p = sph2cart([v_inf, phi_2_p, theta_2_p])

	print("\n\nSolution found with 2 LGA :")
	print("---------------------------")
	print("\nPhi_1(-)   : {}°\nTheta_1(-) : {}°\nPhi_1(+)   : {}°\nTheta_1(+) : {}°".format(phi_1_m*180/np.pi, theta_1_m*180/np.pi,\
	phi_1_p*180/np.pi, theta_1_p*180/np.pi))
	print("\tRotation : {}°".format(np.arccos( np.dot(r_1_m, r_1_p) / np.linalg.norm(r_1_m) / np.linalg.norm(r_1_p) ) * 180/np.pi ))

	print("\nPhi_2(-)   : {}°\nTheta_2(-) : {}°\nPhi_2(+)   : {}°\nTheta_2(+) : {}°".format(phi_1_p*180/np.pi, theta_1_p*180/np.pi,\
	phi_2_p*180/np.pi, theta_2_p*180/np.pi))
	print("\tRotation : {}°".format(np.arccos( np.dot(r_2_p, r_2_m) / np.linalg.norm(r_2_p) / np.linalg.norm(r_2_m) ) * 180/np.pi ))

	t_span_1 = np.array([0, p/q*cst.T_M])
	t_eval_1 = np.linspace(t_span_1[0], t_span_1[-1], 50000)

	t_span_2 = np.array([0, 2*86400])
	t_eval_2 = np.linspace(t_span_2[0], t_span_2[-1], 50000)

	# Computation of the S/C velocity in the Earth inertial frame
	#							 S/C velocity w.r.t the Moon							Moon's velocity
	v_out_1 = P_GEO2HRV(gamma).dot(sph2cart([v_inf, phi_1_p, theta_1_p])) + P_GEO2HRV(gamma).dot(np.array([0, 0, cst.V_M]))
	v_out_2 = P_GEO2HRV(gamma).dot(sph2cart([v_inf, phi_2_p, theta_2_p])) + P_GEO2HRV(gamma).dot(np.array([0, 0, cst.V_M]))

	# Propagation
	sol_1 = solve_ivp(fun=kepler, y0=np.concatenate((r_ar[:3, -1], v_out_1)), t_span=t_span_1, t_eval=t_eval_1, rtol=1e-12, atol=1e-12)
	sol_2 = solve_ivp(fun=kepler, y0=np.concatenate((r_ar[:3, -1], v_out_2)), t_span=t_span_2, t_eval=t_eval_2, rtol=1e-12, atol=1e-12)


	# Rotation of the frame so that at t=tf, the (Ox) ax is following the Earth-Moon vector
	sigma = 2 * np.pi * t_ar[-1] / cst.T_M

	Rot = R2(sigma-gamma)

	for k in range(len(t_ar)):
		r_ar[:3, k] = Rot.dot(r_ar[:3, k])
		r_ar[3:, k] = Rot.dot(r_ar[3:, k])

	for k in range(len(sol_1.t)):
		sol_1.y[:3, k] = Rot.dot(sol_1.y[:3, k])
		sol_1.y[3:, k] = Rot.dot(sol_1.y[3:, k])

	for k in range(len(sol_2.t)):
		sol_2.y[:3, k] = Rot.dot(sol_2.y[:3, k])
		sol_2.y[3:, k] = Rot.dot(sol_2.y[3:, k])


	trajectories = [r_ar, sol_1.y, sol_2.y]
	times = [t_ar, sol_1.t + t_ar[-1], sol_2.t + sol_1.t[-1]]

	return trajectories, times




