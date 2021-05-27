import sys
import numpy as np 
import matplotlib.pyplot as plt
import pickle

from scipy.integrate import solve_ivp, quad, fixed_quad
from scipy.interpolate import interp1d

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.rotation_feasibility import rotation_feasibility
from scripts.earth_departure.resonance_study import resonant_trajectories
from scripts.earth_departure.utils import kepler, cart2sph, sph2cart, R2, P_GEO2HRV, P_HRV2GEO, angle_w_Ox, plot_env_2D, plot_env_3D, thrust_profil_construction
from scripts.earth_departure import constants as cst
from scripts.earth_departure.keplerian_study_results import one_lga, two_lga
from scripts.earth_departure.CR3BP_escape_trajectory import CR3BP_orbit_raising, CR3BP_moon_moon

from scripts.earth_departure.OCP_moon_moon_leg import MoonMoonLeg
from scripts.earth_departure.OCP_apogee_raising import ApogeeRaising
from scripts.earth_departure.OCP_moon_flyby import MoonFlyBy
from collocation.GL_V.src.optimization import Optimization


# Spacecraft characteristics
# --------------------------
Tmax = 2 	 # Maximum thrust [N]
mass = 1000  # Mass			  [kg]

# Trajectory parameters
# ---------------------
eps = 90	  # Thrust arc semi-angle [°]
r_p = 300   # Earth orbit perigee [km]
r_a = 30000   # Earth orbit apogee  [km]

r_m = 300	  # S/C - Moon surface minimal distance [km]

p = 1		# Resonance parameters (Moon) [-]
q = 1		# Resonance parameters (S/C)  [-]

# Outter trajectory characteristics
# ---------------------------------
tau = 15088.095521558473									  		 # Moon departure date (MJD2000)	
v_out = np.array([ 0.84168181508,   0.09065171796, -0.27474864627])  # Velocity at Moon departure in the ECLIPJ2000 frame [km/s]
v_inf = np.linalg.norm(v_out)										 # Excess velocity at Moon departure [km/s]


# 1 - Computation of the Earth - Moon trajectory
# ----------------------------------------------
r_ar, t_ar, thrusts_intervals, last_apogee_pass_time = apogee_raising(mass=mass, T=Tmax/1000, eps=eps*np.pi/180, r_p=r_p, r_a=r_a, v_inf=v_inf)


# 2 - Extraction of informations about the S/C position at Moon encounter
# -----------------------------------------------------------------------
r_in = r_ar[:, -1]			 # Spacecraft state [km] | [km/s]
gamma = angle_w_Ox(r_in[:3]) # Angle between the (Ox) axis and the Spacecraft position w.r.t the Earth [rad]


# 3 - Verification that one LGA (Lunar Gravity Assist) is sufficient 
# ------------------------------------------------------------------
one_lga_, phi_m, theta_m, phi_p, theta_p = rotation_feasibility(v_in=r_in[3:], v_out=v_out, tau=tau, r_m=r_m, gamma=gamma)


if one_lga_ == True:

	# Computation of the Earth - Moon trajectory and the beginning of the departure leg
	trajectories, times = one_lga(v_inf=v_inf, phi_m=phi_m, theta_m=theta_m, phi_p=phi_p, theta_p=theta_p, gamma=gamma, r_ar=r_ar, t_ar=t_ar)
	

	# Pickle of the results in the Keplerian approximation
	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/a_r', 'wb') as f:
		pickle.dump({'trajectory': trajectories[0], 'time': times[0], 'thrusts_intervals': thrusts_intervals, 'mass': mass, \
						'Tmax': Tmax/1000, 'last_apogee_pass_time': last_apogee_pass_time}, f)

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

	# Searching for the possible resonant (p:q) trajectories 
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

		index = int(feasible_index[0])

		_, phi_2_m, theta_2_m, phi_2_p, theta_2_p = rotation_feasibility(v_in=resonant_traj[index, 5:], v_out=v_out, \
			tau=tau, r_m=r_m, gamma=gamma, print_=False)


		# Computation of the Apogee Raising and the Resonant trajectories in the CR3BP frame
		trajectories, times = two_lga(v_inf=v_inf, phi_1_m=phi_1_m, theta_1_m=theta_1_m, phi_1_p=phi_1_p, theta_1_p=theta_1_p, phi_2_m=phi_2_m, theta_2_m=theta_2_m, \
			phi_2_p=phi_2_p, theta_2_p=theta_2_p, p=p, q=q, gamma=gamma, r_ar=r_ar, t_ar=t_ar)

		# Computation of the trajectory in the CR3BP frame
		apogee_raising_cr3bp, apogee_raising_trajectory, apogee_raising_time, \
		fixed_trajectory, fixed_time = CR3BP_orbit_raising(trajectory=trajectories[0], time=times[0], thrusts_intervals=thrusts_intervals, \
		mass=mass, Tmax=Tmax/1000, t_last_ap_pass=last_apogee_pass_time)

		moon_moon_cr3bp, moon_moon_trajectory, moon_moon_time = CR3BP_moon_moon(trajectories[1], times[1])

		# Optimization of the CR3BP Apogee Raising trajectory to make it feasible
		# -----------------------------------------------------------------------
		apogee_raising_problem = ApogeeRaising(apogee_raising_cr3bp, mass, Tmax/1000, apogee_raising_trajectory, apogee_raising_time)

		# Instantiation of the optimization
		optimization = Optimization(problem=apogee_raising_problem)

		# Launch of the optimization
		optimization.run()

		opt_trajectory = optimization.results['opt_st']

		fig = plt.figure()
		ax = fig.gca(projection='3d')

		ax.plot(fixed_trajectory[0], fixed_trajectory[1], fixed_trajectory[2], '-', color='blue', linewidth=1)
		ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], '-', color='blue', linewidth=1)
		ax.plot([-apogee_raising_cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
		ax.plot([1-apogee_raising_cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

		plt.show()

		# Conversion of the Thrust in kN
		optimization.results['opt_ct'][0] *= apogee_raising_cr3bp.T**2 / apogee_raising_cr3bp.L


		with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/a_r_opt', 'wb') as f:
			pickle.dump({'cr3bp': apogee_raising_cr3bp, 'trajectory': optimization.results['opt_st'], 'time': optimization.results['opt_tm'], \
				'thrusts': optimization.results['opt_ct'], 'fixed_trajectory': fixed_trajectory, 'fixed_time': fixed_time, \
				'thrusts_intervals': thrusts_intervals, 'mass': mass, 'Tmax': Tmax/1000}, f)

		# Optimization of the CR3BP Moon-Moon trajectory to make it feasible
		# ------------------------------------------------------------------
		moon_moon_problem = MoonMoonLeg(moon_moon_cr3bp, mass, Tmax/1000, moon_moon_trajectory, moon_moon_time)

		# Instantiation of the optimization
		optimization = Optimization(problem=moon_moon_problem)

		# Launch of the optimization
		optimization.run()

		opt_trajectory = optimization.results['opt_st']

		fig = plt.figure()
		ax = fig.gca(projection='3d')

		ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], '-', color='blue', linewidth=1)
		ax.plot([ -moon_moon_cr3bp.mu], [0], [0], 'o', color='black', markersize=5)
		ax.plot([1-moon_moon_cr3bp.mu], [0], [0], 'o', color='black', markersize=2)

		plt.show()

		with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/moon_moon_opt', 'wb') as f:
			pickle.dump({'cr3bp': moon_moon_cr3bp, 'trajectory': optimization.results['opt_st'], 'time': optimization.results['opt_tm'], \
			 'thrusts': optimization.results['opt_ct'], 'mass': mass, 'Tmax': Tmax/1000}, f)


	else:
		print("No second LGA found with this resonance")



# if __name__ == '__main__':

# 	# Recuperation of the data 
# 	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/a_r_opt', 'rb') as f:
# 		apogee_raising_results = pickle.load(f)

# 	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/moon_moon_opt', 'rb') as f:
# 		moon_moon_results = pickle.load(f)

# 	# Construction of the thrust profil of the apogee raising part
# 	apogee_raising_fixed_thrust = thrust_profil_construction(apogee_raising_results['fixed_time'], apogee_raising_results['fixed_trajectory'], \
# 			apogee_raising_results['thrusts_intervals'], Tmax=apogee_raising_results['Tmax'])

# 	# Dimensionment of thrust and time for the apogee raising phase
# 	apogee_raising_results['thrusts'][0] /= apogee_raising_results['cr3bp'].T**2 / apogee_raising_results['cr3bp'].L
# 	apogee_raising_results['time'] *= apogee_raising_results['cr3bp'].T
# 	apogee_raising_results['fixed_time'] *= apogee_raising_results['cr3bp'].T

# 	apogee_raising_thrust_profil = np.hstack((apogee_raising_fixed_thrust, apogee_raising_results['thrusts']))
# 	apogee_raising_time = np.concatenate((apogee_raising_results['fixed_time'], apogee_raising_results['time']))

# 	# Dimensionment of thrust and time for the moon-moon phase
# 	moon_moon_results['thrusts'][0] /= moon_moon_results['cr3bp'].T**2 / moon_moon_results['cr3bp'].L
# 	moon_moon_results['time'] *= moon_moon_results['cr3bp'].T


# 	# Computation of the orbit raising delta-V [km/s]
# 	interpolation_function = interp1d(apogee_raising_time, apogee_raising_thrust_profil[0] / apogee_raising_results['mass'])
# 	apogee_raising_dV = fixed_quad(func=interpolation_function, a=apogee_raising_time[0], b=apogee_raising_time[-1], n=100)[0]

# 	interpolation_function = interp1d(moon_moon_results['time'], moon_moon_results['thrusts'][0] / apogee_raising_results['mass'])
# 	moon_moon_dV = fixed_quad(func=interpolation_function, a=moon_moon_results['time'][0], b=moon_moon_results['time'][-1], n=100)[0]

# 	print("Total delta-V : {} km/s".format(apogee_raising_dV + moon_moon_dV))


# 	def moon_minimal_distance_reached(t, y):
# 		return np.linalg.norm(np.array([1-0.012151, 0, 0]) - y[:3]) - 1000/384400
# 	moon_minimal_distance_reached.terminal = True

# 	# Computation of the additional legs before and after LGA

# 	# Post-apogee raising
# 	# -------------------
# 	r0_m = apogee_raising_results['trajectory'][:-1, -1]
# 	t_span_m = np.array([0, 10])
# 	t_eval_m = np.linspace(t_span_m[0], t_span_m[-1], 10000)

# 	apogee_raising_final_leg = solve_ivp(fun=apogee_raising_results['cr3bp'].states_derivatives, y0=r0_m, t_span=t_span_m, t_eval=t_eval_m, \
# 		events=(moon_minimal_distance_reached), rtol=1e-12, atol=1e-12)

# 	# Before moon-moon leg
# 	# --------------------
# 	r0_p = moon_moon_results['trajectory'][:-1, 0]
# 	t_span_p = np.array([0, -10])
# 	t_eval_p = np.linspace(t_span_p[0], t_span_p[-1], 10000)

# 	moon_moon_prel_leg = solve_ivp(fun=moon_moon_results['cr3bp'].states_derivatives, y0=r0_p, t_span=t_span_p, t_eval=t_eval_p, \
# 		events=(moon_minimal_distance_reached), rtol=1e-12, atol=1e-12)

# 	# After moon-moon leg
# 	# -------------------
# 	r0_p_2 = moon_moon_results['trajectory'][:-1, -1]
# 	t_span_p_2 = np.array([0, 10])
# 	t_eval_p_2 = np.linspace(t_span_p_2[0], t_span_p_2[-1], 10000)

# 	moon_moon_add_leg = solve_ivp(fun=moon_moon_results['cr3bp'].states_derivatives, y0=r0_p_2, t_span=t_span_p_2, t_eval=t_eval_p_2, \
# 		events=(moon_minimal_distance_reached), rtol=1e-12, atol=1e-12)

# 	apogee_raising_add_trajectory = apogee_raising_final_leg.y
# 	apogee_raising_add_time = apogee_raising_final_leg.t * apogee_raising_results['cr3bp'].T + apogee_raising_time[-1]

# 	moon_moon_prel_trajectory = moon_moon_prel_leg.y
# 	moon_moon_prel_time = -moon_moon_prel_leg.t * moon_moon_results['cr3bp'].T + apogee_raising_add_time[-1]

# 	moon_moon_add_trajectory = moon_moon_add_leg.y
# 	moon_moon_add_time = moon_moon_add_leg.t * moon_moon_results['cr3bp'].T + moon_moon_results['time'][-1]

# 	# Construction of the total time
# 	time_tot = np.concatenate((apogee_raising_time, apogee_raising_add_time, moon_moon_prel_time, moon_moon_results['time'], moon_moon_add_time))

# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)

# 	ax.plot(time_tot)

# 	plt.show()

# 	# Construction of the total trajectory
# 	trajectory_tot = np.hstack((apogee_raising_results['fixed_trajectory'], apogee_raising_results['trajectory'][:-1, :], \
# 		apogee_raising_add_trajectory, moon_moon_prel_trajectory, moon_moon_results['trajectory'][:-1, :], moon_moon_add_trajectory))

# 	# Construction of the total thrust
# 	apogee_raising_add_thrust = np.zeros((4, len(apogee_raising_add_time)))
# 	moon_moon_prel_thrust = np.zeros((4, len(moon_moon_prel_time)))
# 	moon_moon_add_thrust = np.zeros((4, len(moon_moon_add_time)))

# 	thrust_tot = np.hstack((apogee_raising_thrust_profil, apogee_raising_add_thrust, moon_moon_prel_thrust, \
# 		moon_moon_results['thrusts'], moon_moon_add_thrust))

# 	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/results/CD3-2020_earth_departure', 'wb') as file:
# 		pickle.dump({'trajectory': trajectory_tot, 'time': time_tot, 'thrusts': thrust_tot}, file)


# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)

# 	ax.plot(thrust_tot[0])

# 	plt.grid()
# 	plt.show()

# 	fig = plt.figure()
# 	ax = fig.gca(projection='3d')

# 	ax.plot(trajectory_tot[0], trajectory_tot[1], trajectory_tot[2], '-', color='blue', linewidth=1)
# 	ax.plot([1-0.012151], [0], [0], 'o', color='black', markersize=2)
# 	ax.plot([-0.012151], [0], [0], 'o', color='black', markersize=5)

# 	plt.show()








