import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.lga.resonance.utils_2 import kepler_thrust, kep2cart, cart2kep, moon_reached, R2, apside_pass
from scripts.lga.resonance import constants as cst

"""

	Determination of a S/C trajectory between a circular Earth orbit of radius `r_p` and a Moon encounter fixed by a 
	given excess velocity `v_inf`. The Oberth effect is used by igniting the thruster on an arc of angle `eps`. 

	Parameters:
	-----------
		T : [N]
			S/C maximum thrust
		eps : [°]
			Thrust arc semi-angle
		per : [km]
			Earth orbit perigee (relative to the Earth surface)
		v_inf : [km/s]
			Excess velocity required at the Moon encounter

	Returns:
	--------
		r : [km | km/s]
			S/C position and velocity in the Geocentric frame
		t : [s]
			Time grid

"""

def f(r0, t_span, t_eval, T, eps, v_inf):

	sol = solve_ivp(fun=kepler_thrust, y0=r0, t_span=t_span, t_eval=t_eval, args=(T, eps), events=(moon_reached), rtol=1e-12, atol=1e-13)
	r = sol.y

	# Final velocity 
	v_f = sol.y[3:, -1]

	# Angle between (Ox) and the S/C position at t=tf
	theta = np.sign(np.cross(np.array([1, 0, 0]), r[:3, -1])[2]) * np.arccos(np.dot(np.array([1, 0, 0]), r[:3, -1]) / np.linalg.norm(r[:3, -1]))

	# Computation of the velocities difference [km/s]
	v_M = R2(theta).dot(np.array([0, cst.V_M, 0]))
	delta_V = v_f - v_M

	return np.linalg.norm(delta_V) - v_inf


def apogee_raising(T, eps, r_p, v_inf):
	""" 
		Find a trajectory departing from a circular orbit around the Earth, intercepting the Moon
		with a given velocity at infinity.

		Parameters:
		-----------
			T : [kN]
				S/C maximum thrust
			eps : [rad]
				Angle allowing the thrusters ignition or shutdown
			r_p : [km]
				Perigee of the orbit
			v_inf : [km/s]
				Desired velocity at infinity 

	"""


	# 1 - Definition of the initial circular orbit
	# --------------------------------------------
	a  = 2 * (cst.R_E + r_p)    # SMA [km]
	e  = 0                      # Eccentricity [-]
	i  = 0                      # Inclinaison [rad]
	W  = 0                      # RAAN [rad]
	w  = np.pi                  # Perigee anomaly [rad]
	ta = 0                      # True anomaly [rad]


	# 2 - Detemination of the number of revolutions around the Earth before Moon encounter
	# ------------------------------------------------------------------------------------
	r0 = kep2cart(a, e, i, W, w, ta, cst.mu_E) 

	t_span = np.array([0, 200 * 86400])
	t_eval = np.linspace(t_span[0], t_span[-1], 100000)

	moon_reached.terminal = True
	moon_reached.direction = 1

	apside_pass.direction = -1

	sol = solve_ivp(fun=kepler_thrust, y0=r0, t_span=t_span, t_eval=t_eval, args=(T, eps), events=(moon_reached, apside_pass), rtol=1e-12, atol=1e-13)
	r = sol.y

	# Date of the last apogee pass
	last_ap_pass_time = sol.t_events[1][-1]


	# 3 - Computation of the S/C states after the penultimate apogee pass
	# -------------------------------------------------------------------
	t_span = np.array([0, last_ap_pass_time])
	t_eval = np.linspace(t_span[0], t_span[-1], 100000)

	sol1 = solve_ivp(fun=kepler_thrust, y0=r0, t_span=t_span, t_eval=t_eval, args=(T, eps), events=(moon_reached), rtol=1e-12, atol=1e-13)
	r1 = sol1.y
	t1 = sol1.t

	# Last simulation
	t_span = np.array([0, 200 * 86400])
	t_eval = np.linspace(t_span[0], t_span[-1], 100000)

	r0 = r1[:, -1] # New initial conditions are last states of the previous simulation


	# 4 - Computation of the last arc semi-angle to reach the Moon with the desired excess velocity
	# ---------------------------------------------------------------------------------------------
	# Secants method to find the good last thrust arc
	eps0 = eps
	eps1 = eps - 0.5 * np.pi / 180

	f0 = f(r0, t_span, t_eval, T, eps0, v_inf)
	f1 = f(r0, t_span, t_eval, T, eps1, v_inf)

	f2 = 1

	print("Searching for the last Thrust arc angle :\tabsolute error (km/s)\tangle (°)")
	while abs(f2) > 1e-6:
		eps2 = eps1 - (eps1 - eps0) / (f1 - f0) * f1

		f2 = f(r0, t_span, t_eval, T, eps2, v_inf)	

		eps0 = eps1
		f0 = f1

		eps1 = eps2 
		f1 = f2

		print("\t\t\t\t\t\t{}\t\t\t{}".format(round(abs(f2), 5), round(abs(eps2*180/np.pi), 5)))

	# Last epsilon angle [rad]
	eps_l = eps2

	# Simulation of the last branch before Moon encounter
	t_span = np.array([0, 200 * 86400])
	t_eval = np.linspace(t_span[0], t_span[-1], 100000)

	sol2 = solve_ivp(fun=kepler_thrust, y0=r0, t_span=t_span, t_eval=t_eval, args=(T, eps_l), events=(moon_reached), rtol=1e-12, atol=1e-13)
	r2 = sol2.y
	t2 = sol2.t


	# 5 - Construction of the whole trajectory
	# ----------------------------------------

	r = np.concatenate((r1, r2), axis=1)
	t = np.concatenate((t1, t2+last_ap_pass_time))


	# 6 - Plot
	# --------

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot([0], [0], 'o', color='black', markersize=7, label='Earth')
	ax.plot(r[0], r[1], '-', color='blue', linewidth=1, label='S/C trajectory')

	ax.plot( [cst.d_M*np.cos(theta_) for theta_ in np.linspace(0, 2*np.pi, 100)], [cst.d_M*np.sin(theta_) for theta_ in np.linspace(0, 2*np.pi, 100)], \
				'-', color='black', linewidth=1, label='Moon orbit')

	plt.legend()
	plt.grid()
	plt.show()

	print("\n")
	print("S/C Thrust ................. : {} mN".format(T*1e6))
	print("Arc angle .................. : {}°".format(2 * eps * 180 / np.pi))
	print("Last arc angle ............. : {}°".format(2 * eps_l * 180 / np.pi))
	print("Minimal distance to the Moon : {} km".format(r_p+cst.R_M))
	print("\n")
	
	return r, t


if __name__ == '__main__':

	# S/C Thrust [kN]
	T = float(sys.argv[1]) / 1000

	# Thrust arc semi-angle [°]
	eps = float(sys.argv[2]) * np.pi / 180

	# Earth orbit perigee [km]
	r_p = float(sys.argv[3])

	# S/C inbound excess velocity [km/s]
	v_inf = float(sys.argv[4])

	apogee_raising(T=T, eps=eps, r_p=r_p, v_inf=v_inf)

