import sys
import numpy as np 
import matplotlib.pyplot as plt
import pickle

from scipy.integrate import solve_ivp

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.rotation_feasibility import rotation_feasibility
from scripts.earth_departure.resonance_study import resonant_trajectories
from scripts.earth_departure.utils import kepler, cart2sph, sph2cart, R2, P_GEO2HRV, P_HRV2GEO, angle_w_Ox, plot_env_2D, plot_env_3D
from scripts.earth_departure import constants as cst
from scripts.earth_departure.keplerian_study_results import one_lga, two_lga
from scripts.earth_departure.CR3BP_escape_trajectory import CR3BP_orbit_raising, CR3BP_moon_moon

from scripts.earth_departure.OCP_moon_moon_leg import MoonMoonLeg
from scripts.earth_departure.OCP_apogee_raising import ApogeeRaising
from scripts.earth_departure.OCP_moon_flyby import MoonFlyBy
from collocation.GL_V.src.optimization import Optimization


# Spacecraft characteristics
# --------------------------
Tmax = 1 	 # Maximum thrust [N]
mass = 1000  # Mass		      [kg]

# Trajectory parameters
# ---------------------
eps = 90	  # Thrust arc semi-angle [°]
r_p = 30000   # Earth orbit perigee [km]
r_a = 30000   # Earth orbit apogee  [km]

r_m = 3000	  # S/C - Moon surface minimal distance [km]

p = 2		# Resonance parameters (Moon) [-]
q = 1		# Resonance parameters (S/C)  [-]

# Outter trajectory characteristics
# ---------------------------------
tau = 14742.254219498605								      	     # Moon departure date (MJD2000)	
v_out = np.array([-0.10660698523 , 0.56348253138 ,  0.5054158877])   # Velocity at Moon departure in the ECLIPJ2000 frame [km/s]
v_inf = np.linalg.norm(v_out)									     # Excess velocity at Moon departure [km/s]


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

		index = int(feasible_index[-1])

		_, phi_2_m, theta_2_m, phi_2_p, theta_2_p = rotation_feasibility(v_in=resonant_traj[index, 5:], v_out=v_out, \
			tau=tau, r_m=r_m, gamma=gamma, print_=False)


		# Computation of the Apogee Raising and the Resonant trajectories in the CR3BP frame
		trajectories, times = two_lga(v_inf=v_inf, phi_1_m=phi_1_m, theta_1_m=theta_1_m, phi_1_p=phi_1_p, theta_1_p=theta_1_p, phi_2_m=phi_2_m, theta_2_m=theta_2_m, \
			phi_2_p=phi_2_p, theta_2_p=theta_2_p, p=p, q=q, gamma=gamma, r_ar=r_ar, t_ar=t_ar)


		with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/a_r', 'wb') as f:
			pickle.dump({'trajectory': trajectories[0], 'time': times[0], 'thrusts_intervals': thrusts_intervals, 'mass': mass, \
							'Tmax': Tmax/1000, 'last_apogee_pass_time': last_apogee_pass_time}, f)

		with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/moon_moon', 'wb') as f:
			pickle.dump({'trajectory': trajectories[1], 'time': times[1], 'thrusts_intervals': thrusts_intervals, 'mass': mass, \
							'Tmax': Tmax/1000}, f)

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

		with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/a_r_opt', 'wb') as f:
			pickle.dump({'cr3bp': apogee_raising_cr3bp, 'trajectory': optimization.results['opt_st'], 'time': optimization.results['opt_tm'], \
				'fixed_trajectory': fixed_trajectory, 'fixed_time': fixed_time, 'mass': mass, 'Tmax': Tmax/1000}, f)

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
			pickle.dump({'cr3bp': moon_moon_cr3bp, 'trajectory': optimization.results['opt_st'], 'time': optimization.results['opt_tm'], 'mass': mass, \
							'Tmax': Tmax/1000}, f)


	else:
		print("No second LGA found with this resonance")




# if __name__ == '__main__':

# 	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/a_r_opt', 'rb') as f:
# 			res_apogee_raising = pickle.load(f)

# 	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/moon_moon_opt', 'rb') as f:
# 			res_moon_moon = pickle.load(f)

# 	def moon_surface_reached(t, r):
# 		r_m = np.array([1 - 0.012151, 0, 0])
# 		d = np.linalg.norm(r[:3] - r_m) * 384400

# 		min_dist = 1000

# 		return d - min_dist

# 	moon_surface_reached.terminal = True

# 	# Extraction of the cr3bp, trajectories and time grids
# 	# ----------------------------------------------------
# 	apogee_raising_cr3bp = res_apogee_raising['cr3bp']
# 	apogee_raising_trajectory = res_apogee_raising['trajectory']
# 	apogee_raising_time = res_apogee_raising['time']

# 	moon_moon_cr3bp = res_moon_moon['cr3bp']
# 	moon_moon_trajectory = res_moon_moon['trajectory']
# 	moon_moon_time = res_moon_moon['time']

# 	print(np.linalg.norm(apogee_raising_trajectory[3:, -1]))
# 	print(np.linalg.norm(moon_moon_trajectory[3:, 0]))
# 	input()


# 	# Time span and initial condition for forward propagation
# 	# -------------------------------------------------------
# 	t_span_fwd = [0, 30 * 86400 / res_apogee_raising['cr3bp'].T]
# 	t_eval_fwd = np.linspace(t_span_fwd[0], t_span_fwd[-1], 100000)

# 	r0_fwd = apogee_raising_trajectory[:-1, -1]

# 	fwd_solution = solve_ivp(fun=apogee_raising_cr3bp.states_derivatives, y0=r0_fwd, t_span=t_span_fwd, t_eval=t_eval_fwd, \
# 		events=(moon_surface_reached), rtol=1e-12, atol=1e-12)
# 	fwd_trajectory = fwd_solution.y
# 	fwd_time = fwd_solution.t


# 	# Time span and initial condition for backward propagation
# 	# --------------------------------------------------------
# 	t_span_bwd = [30 * 86400 / res_moon_moon['cr3bp'].T, 0]
# 	t_eval_bwd = np.linspace(t_span_bwd[0], t_span_bwd[-1], 100000)

# 	r0_bwd = moon_moon_trajectory[:-1, 0]

# 	bwd_solution = solve_ivp(fun=moon_moon_cr3bp.states_derivatives, y0=r0_bwd, t_span=t_span_bwd, t_eval=t_eval_bwd, \
# 		events=(moon_surface_reached), rtol=1e-12, atol=1e-12)
# 	bwd_trajectory = np.flip(bwd_solution.y, axis=1)
# 	bwd_time = np.flip(bwd_solution.t - bwd_solution.t[-1]) + fwd_time[-1]


# 	moon_flyby_problem = MoonFlyBy(moon_moon_cr3bp, 1000, 10/1000, fwd_trajectory, fwd_time, bwd_trajectory, bwd_time, r_m=5000)

# 	# Instantiation of the optimization
# 	optimization = Optimization(problem=moon_flyby_problem)

# 	# Launch of the optimization
# 	optimization.run()

# 	opt_trajectory = optimization.results['opt_st']

# 	fig = plt.figure()
# 	ax = fig.gca(projection='3d')

# 	ax.plot(opt_trajectory[0], opt_trajectory[1], opt_trajectory[2], \
# 		'-', color='blue', linewidth=1)

# 	ax.plot([1-res_apogee_raising['cr3bp'].mu], [0], [0], 'o', color='black', markersize=2)

# 	plt.show()

	
	

