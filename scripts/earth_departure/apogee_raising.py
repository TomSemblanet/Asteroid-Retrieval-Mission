import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.earth_departure.utils import kepler, kepler_thrust, kep2cart, cart2kep, moon_reached, R2, apside_pass, plot_env_2D, \
												thrust_ignition, angle_w_Ox
from scripts.earth_departure import constants as cst

"""

	Determination of a S/C trajectory between a circular Earth orbit of radius `r_p` and a Moon encounter fixed by a 
	given excess velocity `v_inf`. The Oberth effect is used by igniting the thruster on an arc of angle `eps`. 

	Parameters:
	-----------
		mass : [kg]
			S/C initial mass
		T : [kN]
			S/C maximum thrust
		eps : [°]
			Thrust arc semi-angle
		r_p : [km]
			Earth orbit perigee (relative to the Earth surface)
		v_inf : [km/s]
			Excess velocity required at the Moon encounter

	Returns:
	--------
		r : [km | km/s]
			S/C position and velocity in the Geocentric frame
		t : [s]
			Time grid
		thrust_intervals : [-]
			Array of the thrusts intervals
		last_ap_pass_time : [s]
			Date of last apogee pass

"""

def last_apogee_pass_time(r0, mass, T, eps):
	""" Determines the time of the last S/C apogee pass during its trajectory
		from its initial circular orbit to the Moon encounter.

		Parameters
		----------
			r0 : array
				S/C initial states on its orbit around the Earth [km] | [km/s]
			mass : float
				S/C initial mass [kg]
			T : float
				S/C maximum thrust [kN]
			eps : float
				Angle allowing the thrusters ignition or shutdown [rad]

		Returns
		-------
			tau : float
				Time of the last apogee pass [s]

	""" 

	t_span = np.array([0, 2 * 365 * 86400])
	t_eval = np.linspace(t_span[0], t_span[-1], 100000)

	moon_reached.terminal = True
	moon_reached.direction = 1

	apside_pass.direction = -1

	sol = solve_ivp(fun=kepler_thrust, y0=r0, t_span=t_span, t_eval=t_eval, args=(mass, T, eps), events=(moon_reached, apside_pass, thrust_ignition), \
		rtol=1e-10, atol=1e-13)
	r = sol.y

	# Date of the last apogee pass
	tau = sol.t_events[1][-1]

	# Searching the index of the last pass
	last_pass_index = np.searchsorted(sol.t, tau)

	return sol.y[:, :last_pass_index], sol.t[:last_pass_index], sol.t_events[2][:-2], tau


# def propagate_to_last_apogee_pass(r0, tau, mass, T, eps):
# 	""" Propagation of the keplerian equations from the S/C initial position on its circular orbit around the 
# 		Earth to the last apogee pass before the Moon encounter.

# 		Parameters
# 		----------
# 			r0 : array
# 				S/C initial states on its orbit around the Earth [km] | [km/s]
# 			tau : float
# 				Time of the last apogee pass
# 			mass : float
# 				S/C mass [kg]
# 			T : float
# 				S/C maximum thrust [kN]
# 			eps : float
# 				Angle allowing the thrusters ignition or shutdown [rad]

# 		Returns
# 		-------
# 			r : array
# 				S/C states during the trajectory to the last apogee pass [km] | [km/s]
# 			t : array
# 				Time grid during the trajectory to the last apogee pass [s]
# 			t_ign : array
# 				Dates of thrusters ignition / shut-down [s]

# 	"""

# 	t_span = np.array([0, tau])
# 	t_eval = np.linspace(t_span[0], t_span[-1], 100000)

# 	sol = solve_ivp(fun=kepler_thrust, y0=r0, t_span=t_span, t_eval=t_eval, args=(mass, T, eps), events=(moon_reached, thrust_ignition), rtol=1e-10, atol=1e-13)
	
# 	return sol.y, sol.t, sol.t_events[1]


def last_arc_search(r_ap, v_inf, mass, T, eps, eps_guess):
	""" Search the last thrust arc length to encounter the Moon with the desired excess velocity using the secants
		method.

		Parameters
		----------
			r_ap : array
				S/C states at the last apogee pass [km] | [km/s]
			v_inf : float
				S/C excess velocity at Moon encounter [km/s]
			mass : float
				S/C initial mass [kg]
			T : float
				S/C maximum thrust [kN]
			eps : float
				Angle allowing the thrusters ignition or shutdown [rad]

		Returns
		-------
			eps_l : float
				Last thrust arc length [rad]

	"""

	t_span = np.array([0, 2 * 365 * 86400])
	t_eval = np.linspace(t_span[0], t_span[-1], 100000)

	# The propagation begins at the last apogee pass
	r0 = r_ap[:, -1]

	# Secants method to find the good last thrust arc
	eps0 = eps_guess
	eps1 = eps_guess - 0.5 * np.pi / 180

	f0 = f(r0, t_span, t_eval, mass, T, eps0, v_inf)
	f1 = f(r0, t_span, t_eval, mass, T, eps1, v_inf)
	f2 = 1

	error = False

	print("Searching for the last Thrust arc angle :\tabsolute error (km/s)\tangle (°)", flush=True)
	while abs(f2) > 1e-3:
		eps2 = eps1 - (eps1 - eps0) / (f1 - f0) * f1

		f2 = f(r0, t_span, t_eval, mass, T, eps2, v_inf)	

		eps0 = eps1
		f0 = f1

		eps1 = eps2 
		f1 = f2

		print("\t\t\t\t\t\t{}\t\t\t{}".format(round(abs(f2), 5), round(abs(eps2*180/np.pi), 5)), flush=True)

		if (str(eps2) == 'nan' or str(eps2) == 'inf'):
			error = True
			break

	return eps2, error


def f(r0, t_span, t_eval, mass, T, eps, v_inf):

	sol = solve_ivp(fun=kepler_thrust, y0=r0, t_span=t_span, t_eval=t_eval, args=(mass, T, eps), events=(moon_reached), rtol=1e-12, atol=1e-12)
	r = sol.y

	# Final velocity 
	v_f = sol.y[3:, -1]

	# Angle between (Ox) and the S/C position at t=tf
	theta = angle_w_Ox(sol.y[:3, -1])

	# Computation of the velocities difference [km/s]
	v_M = R2(theta).dot(np.array([0, cst.V_M, 0]))
	delta_V = v_f - v_M

	return np.linalg.norm(delta_V) - v_inf


def propagate_last_branch(r_ap, mass, T, eps_l):
	""" Search the last thrust arc length to encounter the Moon with the desired excess velocity using the secants
		method.

		Parameters
		----------
			r_ap : array
				S/C states at the last apogee pass [km] | [km/s]
			mass : float
				S/C mass [kg]
			T : float
				S/C maximum thrust [kN]
			eps : float
				Last angle allowing the thrusters ignition or shutdown [rad]

		Returns
		-------
			r : array
				S/C states during the trajectory after the last apogee pass [km] | [km/s]
			t : array
				Time grid during the trajectory after the last apogee pass [s]
			t_ign : array
				Dates of thrusters ignition / shut-down

	"""

	t_span = np.array([0, 2 * 365 * 86400])
	t_eval = np.linspace(t_span[0], t_span[-1], 100000)

	sol = solve_ivp(fun=kepler_thrust, y0=r_ap[:, -1], t_span=t_span, t_eval=t_eval, args=(mass, T, eps_l), events=(moon_reached, thrust_ignition), rtol=1e-12, atol=1e-12)

	return sol.y, sol.t, sol.t_events[1]


def apogee_raising(mass, T, eps, r_p, r_a, v_inf):
	""" 
		Find a trajectory departing from a circular orbit around the Earth, intercepting the Moon
		with a given velocity at infinity.

		Parameters:
		-----------
			mass : [kg]
				S/C mass
			T : [kN]
				S/C maximum thrust
			eps : [rad]
				Angle allowing the thrusters ignition or shutdown
			r_p : [km]
				Perigee of the orbit
			r_a : [km]
				Apogee of the orbit
			v_inf : [km/s]
				Desired velocity at infinity 

	"""


	# 1 - Definition of the initial circular orbit and S/C initial state in the Earth inertial frame
	# ----------------------------------------------------------------------------------------------
	a  = (2*cst.R_E + r_p + r_a) / 2		 # SMA [km]
	e  = 1 - (cst.R_E + r_p) / a			 # Eccentricity [-]
	i  = 0							 	 	 # Inclinaison [rad]
	W  = 0				 				 	 # RAAN [rad]
	w  = np.pi			 				 	 # Perigee anomaly [rad]
	ta = - eps				  				 # True anomaly [rad]

	# S/C initial states on its circular orbit around the Earth [km] | [km/s]
	r0 = kep2cart(a, e, i, W, w, ta, cst.mu_E)


	# # 2 - Detemination of the last apogee pass date to begin the last thrust arc length
	# # ---------------------------------------------------------------------------------
	# last_ap_pass_time = last_apogee_pass_time(r0=r0, mass=mass, T=T, eps=eps)


	# # 3 - Computation of the S/C states after the last apogee pass date
	# # -----------------------------------------------------------------
	# r_ap, t_ap, t_thrusters_ap = propagate_to_last_apogee_pass(r0=r0, tau=last_ap_pass_time, mass=mass, T=T, eps=eps)

	r_ap, t_ap, t_thrusters_ap, last_ap_pass_time = last_apogee_pass_time(r0=r0, mass=mass, T=T, eps=eps)


	# 4 - Computation of the last arc semi-angle to reach the Moon with the desired excess velocity
	# ---------------------------------------------------------------------------------------------
	error = True 
	eps_guess = 1 * np.pi / 180
	while error == True:
		eps_l, error = last_arc_search(r_ap=r_ap, v_inf=v_inf, mass=mass, T=T, eps=eps, eps_guess=eps_guess)
		eps_guess += 1 * np.pi / 180


	# 5 - Simulation of the last branch until Moon encounter
	# ------------------------------------------------------
	r_lb, t_lb, t_thrusters_lb = propagate_last_branch(r_ap=r_ap, mass=mass, T=T, eps_l=eps_l)

	# 6 - Construction of the whole trajectory
	# ----------------------------------------
	r = np.concatenate((r_ap, r_lb), axis=1)
	t = np.concatenate((t_ap, t_lb + last_ap_pass_time))

	t_thrusters = np.concatenate((t_thrusters_ap, t_thrusters_lb + last_ap_pass_time))
	thrust_intervals = np.reshape(a=t_thrusters, newshape=(int(len(t_thrusters) / 2), 2))

	# 8 - Plot
	# --------
	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(r[0], r[1], '-', color='blue', linewidth=1, label='S/C trajectory')

	plot_env_2D(ax=ax)
	
	plt.legend()
	plt.grid()
	plt.show()


	# 9 - Informations display
	# ------------------------

	print("\n")
	print("S/C Mass ................... : {} kg".format(mass))
	print("S/C Thrust ................. : {} mN".format(T*1e6))
	print("Arc angle .................. : {}°".format(2 * eps * 180 / np.pi))
	print("Last arc angle ............. : {}°".format(2 * eps_l * 180 / np.pi))
	print("Minimal distance to the Earth : {} km".format(r_p+cst.R_M))
	print("\n")
	
	return r, t, thrust_intervals, last_ap_pass_time

