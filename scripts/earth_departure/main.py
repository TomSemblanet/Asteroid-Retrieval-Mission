import sys
import numpy as np 
import matplotlib.pyplot as plt
import pickle

from scipy.integrate import solve_ivp, quad, fixed_quad
from scipy.interpolate import interp1d

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.rotation_feasibility import rotation_feasibility
from scripts.earth_departure.resonance_study import resonant_trajectories
from scripts.earth_departure.utils import kepler, cart2sph, sph2cart, R2, P_GEO2HRV, P_HRV2GEO, angle_w_Ox, plot_env_2D, plot_env_3D, \
											thrust_profil_construction, plot_earth_departure

from scripts.earth_departure import constants as cst
from scripts.earth_departure.keplerian_study_results import one_lga, two_lga
# from scripts.earth_departure.CR3BP_escape_trajectory import CR3BP_orbit_raising, CR3BP_moon_moon
# from scripts.earth_departure.TBP_apogee_raising import TBP_apogee_raising

# from scripts.earth_departure.OCP_moon_moon_leg import MoonMoonLeg
# from scripts.earth_departure.OCP_apogee_raising import ApogeeRaising
# from scripts.earth_departure.OCP_moon_flyby import MoonFlyBy
# from collocation.GL_V.src.optimization import Optimization


# Thrusters : 2 x EP-THR-2500/1000/20000
#				 	- 40   kW					Total Power
#					- 2000 mN 				    Total Thrust
#					- 2500 s 					Specific Impulse

file_path = sys.argv[1]

with open(file_path, 'rb') as file:
	Earth_NEA = pickle.load(file)

departure_date = Earth_NEA['population'].get_x()[0][0] - Earth_NEA['population'].get_x()[0][1]
excess_velocity_vec = Earth_NEA['udp'].vinf_max * Earth_NEA['population'].get_x()[0][3:6]


# 1 - Spacecraft characteristics definition
# -----------------------------------------
Tmax = 2 	 # Maximum thrust [N]
mass = 10000 # Mass			  [kg]

# 2 - Trajectory parameters definition
# ------------------------------------
eps = 130	  # Thrust arc semi-angle [°]
r_p = 10000     # Earth orbit perigee [km]
r_a = 10000   # Earth orbit apogee  [km]

r_m = 300	  # S/C - Moon surface minimal distance [km]

p = 1		# Resonance parameters (Moon) [-]
q = 1		# Resonance parameters (S/C)  [-]

# 3 - Loading of the outter trajectory characteristics
# ----------------------------------------------------
tau = departure_date						# Moon departure date (MJD2000)	
v_out = excess_velocity_vec / 1000  		# Velocity at Moon departure in the ECLIPJ2000 frame [km/s]
v_inf = np.linalg.norm(v_out)				# Excess velocity at Moon departure [km/s]

# 4 - Computation of the Earth - Moon trajectory
# ----------------------------------------------
r_ar, t_ar, thrusts_intervals, last_apogee_pass_time = apogee_raising(mass=mass, T=Tmax/1000, eps=eps*np.pi/180, r_p=r_p, r_a=r_a, v_inf=v_inf)

time_sum = 0
for interv in thrusts_intervals:
	time_sum += interv[1] - interv[0]

print("Total thrust time: {} sec".format(time_sum))
print("Delta V : {} km/s".format(Tmax * time_sum / mass / 1000))

# 5 - Extraction of informations about the S/C position at Moon encounter
# -----------------------------------------------------------------------
r_in = r_ar[:, -1]			 # Spacecraft state [km] | [km/s]
gamma = angle_w_Ox(r_in[:3]) # Angle between the (Ox) axis and the Spacecraft position w.r.t the Earth [rad]


# 6 - Check if one LGA (Lunar Gravity Assist) is sufficient 
# ---------------------------------------------------------
one_lga_, phi_m, theta_m, phi_p, theta_p = rotation_feasibility(v_in=r_in[3:], v_out=v_out, tau=tau, r_m=r_m, gamma=gamma)


if one_lga_ == True:

	# Computation of the Earth - Moon trajectory and the beginning of the departure leg
	trajectories, times = one_lga(v_inf=v_inf, phi_m=phi_m, theta_m=theta_m, phi_p=phi_p, theta_p=theta_p, gamma=gamma, r_ar=r_ar, t_ar=t_ar)
	

	# # Pickle of the results in the Keplerian approximation
	# with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/a_r', 'wb') as f:
	# 	pickle.dump({'trajectory': trajectories[0], 'time': times[0], 'thrusts_intervals': thrusts_intervals, 'mass': mass, \
	# 					'Tmax': Tmax/1000, 'last_apogee_pass_time': last_apogee_pass_time}, f)

	# Computation of the trajectory in the CR3BP frame
	cr3bp, trajectory, time = CR3BP_orbit_raising(trajectory=trajectories[0], time=times[0], thrusts_intervals=thrusts_intervals, \
		mass=mass, Tmax=Tmax/1000, t_last_ap_pass=last_apogee_pass_time)

	# Optimization of the CR3BP trajectory to make it feasible
	# --------------------------------------------------------
	problem = ApogeeRaising(cr3bp, mass, Tmax/1000, trajectory, time)

	# Instantiation of the optimization
	optimization = Optimization(problem=problem)

	# Launch of the optimization
	optimization.run()

	opt_trajectory = optimization.results['opt_st']
	opt_controls = optimization.results['opt_ct']
	opt_time = optimization.results['opt_tm']

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], '-', color='blue', linewidth=1)
	ax.plot([-cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
	ax.plot([1-cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

	plt.show()



else:

	# We'll include one additional rotation corresponding to the second LGA
	phi_1_m, theta_1_m = phi_m, theta_m
	phi_2_p, theta_2_p = phi_p, theta_p
	phi_1_p, theta_1_p = None, None
	phi_2_m, theta_2_m = None, None

	scd_lga_fnd = False

	# 7 - Searching for the possible resonant (p:q) trajectories 
	# ----------------------------------------------------------
	resonant_traj = resonant_trajectories(v_inf_mag=v_inf, phi_m=phi_1_m, theta_m=theta_1_m, phi_p=phi_2_p, theta_p=theta_2_p, \
		r_m=r_m, gamma=gamma, p=p, q=q)

	feasible_index = np.empty((0))

	for k, traj in enumerate(resonant_traj):

		# Moon departure angles [-]
		phi_1_p, theta_1_p = traj[:2]

		# Check if the trajectory allow to escape the Earth-Moon system with the good velocity
		scd_lga, phi_2_m, theta_2_m, phi_2_p, theta_2_p = rotation_feasibility(v_in=traj[5:], v_out=v_out, tau=tau, r_m=r_m, gamma=gamma, print_=False)
		
		if scd_lga == True: 
			feasible_index = np.append(feasible_index, k)
			scd_lga_fnd = True


	if scd_lga_fnd == True:

		index = int(feasible_index[-1])

		_, phi_2_m, theta_2_m, phi_2_p, theta_2_p = rotation_feasibility(v_in=resonant_traj[index, 5:], v_out=v_out, \
			tau=tau, r_m=r_m, gamma=gamma, print_=False)


		# Computation of the Apogee Raising and the Resonant trajectories in the CR3BP frame
		trajectories, times = two_lga(v_inf=v_inf, phi_1_m=phi_1_m, theta_1_m=theta_1_m, phi_1_p=phi_1_p, theta_1_p=theta_1_p, phi_2_m=phi_2_m, theta_2_m=theta_2_m, \
			phi_2_p=phi_2_p, theta_2_p=theta_2_p, p=p, q=q, gamma=gamma, r_ar=r_ar, t_ar=t_ar)



		# # 8 - Computation of the Moon-Moon leg trajectory in the CR3BP frame
		# # --------------------------------------------------------------
		# moon_moon_cr3bp, moon_moon_trajectory, moon_moon_time = CR3BP_moon_moon(trajectories[1].copy(), times[1].copy())

		# # 9 - Optimization of the CR3BP Moon-Moon trajectory to make it feasible
		# # ----------------------------------------------------------------------
		# moon_moon_problem = MoonMoonLeg(moon_moon_cr3bp, mass, Tmax/1000, moon_moon_trajectory, moon_moon_time)

		# # Instantiation of the optimization
		# optimization = Optimization(problem=moon_moon_problem)

		# # Launch of the optimization
		# optimization.run()

		# opt_trajectory = optimization.results['opt_st']
		# opt_controls = optimization.results['opt_ct']
		# opt_time = optimization.results['opt_tm']

		# fig = plt.figure()
		# ax = fig.gca(projection='3d')

		# ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], '-', color='blue', linewidth=1)
		# ax.plot([ -moon_moon_cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
		# ax.plot([1-moon_moon_cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

		# plt.show()


		# # 10 - Conversion of the moon-moon leg states in the ECI frame
		# # ------------------------------------------------------------

		# moon_moon_trajectory = np.zeros(shape=(6, opt_trajectory.shape[1]))
		# moon_moon_time = opt_time * moon_moon_cr3bp.T
		# moon_moon_thrust = opt_controls[0] * moon_moon_cr3bp.L / moon_moon_cr3bp.T**2

		# for k, t in enumerate(opt_time):
		# 	moon_moon_trajectory[:, k] = moon_moon_cr3bp.syn2eci(t, opt_trajectory[:-1, k])
		# 	moon_moon_trajectory[:3, k] *= moon_moon_cr3bp.L
		# 	moon_moon_trajectory[3:, k] *= moon_moon_cr3bp.V

		fig = plt.figure()
		ax = fig.gca(projection='3d')

		ax.plot(trajectories[0][0], trajectories[0][1], trajectories[0][2], '-', color='blue', linewidth=1)
		ax.plot(trajectories[1][0], trajectories[1][1], trajectories[1][2], '-', color='red', linewidth=1)

		plot_env_3D(ax)
		plt.show()


		# # 11 - Construction of the Moon-Moon leg thrust intervals
		# # -------------------------------------------------------
		moon_moon_thrust_intervals = list()

		# thrust_ON = False
		# t_i, t_f = 0, 0

		# for k, t in enumerate(moon_moon_time):
		# 	if (moon_moon_thrust[k] > 1e-3 and thrust_ON == False):
		# 		thrust_ON = True
		# 		t_i = t

		# 	elif (moon_moon_thrust[k] < 1e-3 and thrust_ON == True):
		# 		thrust_ON = False
		# 		t_f = t
		# 		moon_moon_thrust_intervals.append([t_i, t_f])

		# 	elif (t == moon_moon_time[-1] and thrust_ON == True):
		# 		thrust_ON = False
		# 		t_f = t
		# 		moon_moon_thrust_intervals.append([t_i, t_f])

		# moon_moon_thrust_intervals = np.array(moon_moon_thrust_intervals)

		# 12 - Pickle of the results
		# --------------------------
		with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/results/earth_departure', 'wb') as file:
			pickle.dump({'apogee_raising_trajectory': trajectories[0], 'apogee_raising_time': times[0], 'apogee_raising_thrust_intervals':thrusts_intervals, \
					    'moon_moon_trajectory': trajectories[1], 'moon_moon_time': times[1], 'moon_moon_thrust_intervals': moon_moon_thrust_intervals, \
					    'outter_trajectory': trajectories[2], 'outter_time': times[2], 'Tmax': Tmax}, file)


	else:
		print("No second LGA found with this resonance", flush=True)






if __name__ == '__main__':


	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/results/earth_departure', 'rb') as file:
			results = pickle.load(file)

	apogee_raising_trajectory, apogee_raising_time, apogee_raising_thrust_intervals, moon_moon_trajectory, \
		moon_moon_time, moon_moon_thrust_intervals, outter_trajectory, outter_time, Tmax = results.values()

	plot_earth_departure(apogee_raising_trajectory, apogee_raising_time, apogee_raising_thrust_intervals, \
						 moon_moon_trajectory, moon_moon_time, moon_moon_thrust_intervals, \
						 outter_trajectory, outter_time, Tmax)
