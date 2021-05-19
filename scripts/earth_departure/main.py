import sys
import numpy as np 
import matplotlib.pyplot as plt
import pickle

from scipy.integrate import solve_ivp

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.rotation_feasibility import rotation_feasibility
from scripts.earth_departure.resonance_study import resonant_trajectories
from scripts.earth_departure.utils import kepler, cart2sph, sph2cart, P_GEO2HRV, P_HRV2GEO, angle_w_Ox, plot_env_3D
from scripts.earth_departure import constants as cst
from scripts.earth_departure.keplerian_study_results import one_lga, two_lga
from scripts.earth_departure.CR3BP_escape_trajectory import CR3BP_orbit_raising, CR3BP_moon_moon



# Spacecraft maximum thrust [N]
Tmax = 10

# Spacecraft mass [kg]
mass = 1000

# Thrust arc semi-angle [°]
eps = 90	

# Earth orbit perigee [km]
r_p = 20000

# Resonance parameters [-]
p = 1
q = 1

# Extraction of the S/C excess velocity at Moon departure in the ECLIPJ200 frame [km/s]
v_out = np.array([-0.67827715326,  0.63981981778, -0.23054482431])

# Moon departure date (mj2000)	
tau = 15128.755128883051

# S/C - Moon surface minimal distance [km]
r_m = 5000

# Excess velocity at Moon encounter [km/s]
v_inf = np.linalg.norm(v_out)


# Earth circular orbit -> Moon encounter trajectory
r_ar, t_ar, thrusts_intervals = apogee_raising(mass=mass, T=Tmax/1000, eps=eps*np.pi/180, r_p=r_p, v_inf=v_inf)

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
	
	# Apogee raising 
	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/a_r', 'wb') as f:
		pickle.dump({'trajectory': trajectories[0], 'time': times[0], 'thrusts_intervals': thrusts_intervals, 'mass': mass, \
						'Tmax': Tmax/1000}, f)

	CR3BP_orbit_raising(trajectory=trajectories[0], time=times[0], thrusts_intervals=thrusts_intervals, mass=mass, Tmax=Tmax/1000)


else:

	# We'll include one additional rotation corresponding to the second LGA
	phi_1_m, theta_1_m = phi_m, theta_m
	phi_2_p, theta_2_p = phi_p, theta_p

	phi_1_p, theta_1_p = None, None
	phi_2_m, theta_2_m = None, None

	r_fs = resonant_trajectories(v_inf_mag=v_inf, phi_m=phi_1_m, theta_m=theta_1_m, phi_p=phi_2_p, theta_p=theta_2_p, \
		r_m=r_m, gamma=gamma, p=p, q=q)

	for r_f in r_fs:

		phi_1_p, theta_1_p = r_f[:2]
		scd_lga, phi_2_m, theta_2_m, phi_2_p, theta_2_p = rotation_feasibility(v_in=r_f[5:], v_out=v_out, tau=tau, r_m=r_m, gamma=gamma, print_=False)

		if scd_lga == True: 
			break


	if scd_lga == True:

		trajectories, times = two_lga(v_inf=v_inf, phi_1_m=phi_1_m, theta_1_m=theta_1_m, phi_1_p=phi_1_p, theta_1_p=theta_1_p, phi_2_m=phi_2_m, theta_2_m=theta_2_m, \
			phi_2_p=phi_2_p, theta_2_p=theta_2_p, p=p, q=q, gamma=gamma, r_ar=r_ar, t_ar=t_ar)

		with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/a_r', 'wb') as f:
			pickle.dump({'trajectory': trajectories[0], 'time': times[0], 'thrusts_intervals': thrusts_intervals, 'mass': mass, \
							'Tmax': Tmax/1000}, f)

		with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/moon_moon', 'wb') as f:
			pickle.dump({'trajectory': trajectories[1], 'time': times[1], 'thrusts_intervals': thrusts_intervals, 'mass': mass, \
							'Tmax': Tmax/1000}, f)

		CR3BP_moon_moon(trajectories[1], times[1])

	else:
		print("No second LGA found with this resonance")


