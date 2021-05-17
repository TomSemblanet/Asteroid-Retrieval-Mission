import sys
import numpy as np 
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.rotation_feasibility import rotation_feasibility
from scripts.earth_departure.resonance_study import resonant_trajectories
from scripts.earth_departure.utils import kepler, cart2sph, sph2cart, P_GEO2HRV, P_HRV2GEO, angle_w_Ox
from scripts.earth_departure import constants as cst
from scripts.earth_departure.keplerian_study_results import one_lga
from scripts.earth_departure.GEO_ECLJ2000_transfer import ECLJ2000_trajectory



# Spacecraft maximum thrust [N]
T = 1

# Thrust arc semi-angle [°]
eps = 2

# Earth orbit perigee [km]
r_p = 200

# Resonance parameters [-]
p = 1
q = 1

# Extraction of the S/C excess velocity at Moon departure in the ECLIPJ200 frame [km/s]
v_out = np.array([-0.67827715326,  0.63981981778, -0.23054482431])

# Moon departure date (mj2000)
tau = 15128.755128883051

# S/C - Moon surface minimal distance [km]
r_m = 200

# Excess velocity at Moon encounter [km/s]
v_inf = np.linalg.norm(v_out)


# Earth circular orbit -> Moon encounter trajectory
r_ar, t_ar = apogee_raising(T=T/1000, eps=eps*np.pi/180, r_p=r_p, v_inf=v_inf)

# Extraction of the S/C position and velocity at Moon encounter [km] | [km/s]
r_in = r_ar[:, -1]


# Angle between the (Ox) axe and the S/C position w.r.t the Earth
gamma = angle_w_Ox(r_in[:3])


# Check if one LGA is sufficient to rotate the excess velocity vector
phi_m, theta_m = None, None   # Excess velocity spherical coordinates in the HRV frame before the LGA
phi_p, theta_p = None, None   # Excess velocity spherical coordinates in the HRV frame after the LGA

one_lga_, phi_m, theta_m, phi_p, theta_p = rotation_feasibility(v_in=r_in[3:], v_out=v_out, tau=tau, r_m=r_m, gamma=gamma)


if one_lga_ == True:
	trajectories, times = one_lga(v_inf=v_inf, phi_m=phi_m, theta_m=theta_m, phi_p=phi_p, theta_p=theta_p, gamma=gamma, r_ar=r_ar, t_ar=t_ar)
	ECLJ2000_trajectory(tau=tau, trajectories=trajectories, times=times)

else:

	# We'll include one additional rotation corresponding to the second LGA
	phi_1_m, theta_1_m = phi_m, theta_m
	phi_2_p, theta_2_p = phi_p, theta_p

	phi_1_p, theta_1_p = None, None
	phi_2_m, theta_2_m = None, None

	scd_lga = False

	r_fs = resonant_trajectories(v_inf_mag=v_inf, phi_m=phi_1_m, theta_m=theta_1_m, phi_p=phi_2_p, theta_p=theta_2_p, \
		r_m=r_m, gamma=gamma, p=p, q=q	)

	for r_f in r_fs:

		phi_1_p, theta_1_p = r_f[:2]
		scd_lga, phi_2_m, theta_2_m, phi_2_p, theta_2_p = rotation_feasibility(v_in=r_f[5:], v_out=v_out, tau=tau, r_m=r_m, gamma=gamma, print_=False)

		if scd_lga == True: break


	if scd_lga == True:
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



		# Propagation and plot
		# --------------------

		t_span_1 = np.array([0, p/q*cst.T_M])
		t_span_2 = np.array([0, 2*86400])

		# Computation of the S/C velocity in the Earth inertial frame
		#                             S/C velocity w.r.t the Moon                            Moon's velocity
		v_out_1 = P_GEO2HRV(gamma).dot(sph2cart([v_inf, phi_1_p, theta_1_p])) + P_GEO2HRV(gamma).dot(np.array([0, 0, cst.V_M]))
		v_out_2 = P_GEO2HRV(gamma).dot(sph2cart([v_inf, phi_2_p, theta_2_p])) + P_GEO2HRV(gamma).dot(np.array([0, 0, cst.V_M]))

		sol_1 = solve_ivp(fun=kepler, y0=np.concatenate((r_in[:3], v_out_1)), t_span=t_span_1, rtol=1e-12, atol=1e-12)
		sol_2 = solve_ivp(fun=kepler, y0=np.concatenate((r_in[:3], v_out_2)), t_span=t_span_2, rtol=1e-12, atol=1e-12)

		fig = plt.figure()
		ax = fig.gca(projection='3d')

		ax.plot([0], [0], [0], 'o', markersize=8, color='black')

		ax.plot(r_ar[0], r_ar[1], r_ar[2], '-', linewidth=1, color='blue')
		ax.plot(sol_1.y[0], sol_1.y[1], sol_1.y[2], '-', linewidth=1, color='blue')
		ax.plot(sol_2.y[0], sol_2.y[1], sol_2.y[2], '-', linewidth=1, color='blue')

		ax.plot(np.array([cst.d_M*np.cos(t) for t in np.linspace(0, 2*np.pi, 100)]), np.array([cst.d_M*np.sin(t) for t in np.linspace(0, 2*np.pi, 100)]), \
				np.zeros(100), '-', linewidth=1, color='black')

		plt.show()

	else:
		print("No second LGA found with this resonance")


