import sys
import numpy as np 
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.lga.resonance.apogee_raising import apogee_raising
from scripts.lga.resonance.earth_moon_escape import rotation_feasibility
from scripts.lga.resonance.resonance_study import resonant_trajectories
from scripts.lga.resonance.utils_2 import kepler, cart2sph, sph2cart, P_GEO2HRV
from scripts.lga.resonance import constants as cst

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Spacecraft maximum thrust [N]
T = 1

# Thrust arc semi-angle [°]
eps = 2

# Earth orbit perigee [km]
r_p = 200

# Excess velocity at Moon encounter [km/s]
v_inf = 0.960514


r_ar, t_ar = apogee_raising(T=T/1000, eps=eps*np.pi/180, r_p=r_p, v_inf=v_inf)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Extraction of the S/C position and velocity at Moon encounter [km] | [km/s]
r_in = r_ar[:, -1]

# Angle between the (Ox) axe and the S/C position w.r.t the Earth
gamma = np.sign(np.cross(np.array([1, 0, 0]), r_in[:3])[2]) * np.arccos( np.dot(np.array([1, 0, 0]), r_in[:3]) / np.linalg.norm(r_in[:3]) )

# Extraction of the S/C excess velocity at Moon departure in the ECLIPJ200 frame [km/s]
v_out = np.array([-0.67827715326,  0.63981981778, -0.23054482431])

# Moon departure date (mj2000)
tau = 15128.755128883051

# S/C - Moon minimal distance [km]
r_m = 2000

one_lga, phi_1_m, theta_1_m, phi_2_p, theta_2_p = rotation_feasibility(r_in=r_in, v_out=v_out, tau=tau, r_m=r_m)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if one_lga == True:
	print("\nPhi(-)   : {}°\nTheta(-) : {}°\nPhi(+)   : {}°\nTheta(+) : {}°".format(phi_1_m*180/np.pi, theta_1_m*180/np.pi,\
		phi_2_p*180/np.pi, theta_2_p*180/np.pi))

else:
	r_fs = resonant_trajectories(v_inf_mag=v_inf, phi_m=phi_1_m, theta_m=theta_1_m, phi_p=phi_2_p, theta_p=theta_2_p, \
		r_m=r_m, gamma=gamma, p=2, q=1)

	scd_lga = False
	phi_1_p, theta_1_p = None, None
	phi_2_m, theta_2_m = None, None

	for r_f in r_fs:

		phi_1_p, theta_1_p = r_f[:2]
		scd_lga, phi_2_m, theta_2_m, phi_2_p, theta_2_p = rotation_feasibility(r_in=r_f[2:], v_out=v_out, tau=tau, r_m=r_m)

		if scd_lga == True:
			break

	if scd_lga == True:
		print("Solution found with 2 LGA :")
		print("---------------------------")
		print("\nPhi_1(-)   : {}°\nTheta_1(-) : {}°\nPhi_1(+)   : {}°\nTheta_1(+) : {}°".format(phi_1_m*180/np.pi, theta_1_m*180/np.pi,\
		phi_1_p*180/np.pi, theta_1_p*180/np.pi))
		print("\nPhi_2(-)   : {}°\nTheta_2(-) : {}°\nPhi_2(+)   : {}°\nTheta_2(+) : {}°".format(phi_1_p*180/np.pi, theta_1_p*180/np.pi,\
		phi_2_p*180/np.pi, theta_2_p*180/np.pi))

		t_span_1 = np.array([0, 60*86400])
		t_span_2 = np.array([0, 1*86400])

		# Computation of the S/C velocity in the Earth inertial frame
		#                             S/C velocity w.r.t the Moon                            Moon's velocity
		v_out_1 = P_GEO2HRV(gamma).dot(sph2cart([v_inf, phi_1_p, theta_1_p])) + P_GEO2HRV(gamma).dot(np.array([0, 0, cst.V_M]))
		v_out_2 = P_GEO2HRV(gamma).dot(sph2cart([v_inf, phi_2_p, theta_2_p])) + P_GEO2HRV(gamma).dot(np.array([0, 0, cst.V_M]))

		sol_1 = solve_ivp(fun=kepler, y0=np.concatenate((r_in[:3], v_out_1)), t_span=t_span_1, rtol=1e-12, atol=1e-12)
		sol_2 = solve_ivp(fun=kepler, y0=np.concatenate((r_in[:3], v_out_2)), t_span=t_span_2, rtol=1e-12, atol=1e-12)

		fig = plt.figure()
		ax = fig.gca(projection='3d')

		ax.plot([0], [0], [0], 'o', markersize=8, color='black')

		ax.plot(r_ar[0], r_ar[1], r_ar[2], '-', linewidth=1, color='green')
		ax.plot(sol_1.y[0], sol_1.y[1], sol_1.y[2], '-', linewidth=1, color='blue')
		ax.plot(sol_2.y[0], sol_2.y[1], sol_2.y[2], '-', linewidth=1, color='red')

		ax.plot(np.array([cst.d_M*np.cos(t) for t in np.linspace(0, 2*np.pi, 100)]), np.array([cst.d_M*np.sin(t) for t in np.linspace(0, 2*np.pi, 100)]), \
				np.zeros(100), '-', linewidth=1, color='black')

		plt.show()

	else:
		print("No second LGA found with this resonance")


