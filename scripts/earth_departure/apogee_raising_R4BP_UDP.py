import matplotlib.pyplot as plt 

import sys
import pykep as pk 
import pygmo as pg 
import numpy as np

import pickle 

from scipy.integrate import solve_ivp

from scripts.utils import load_sqp, load_kernels, load_bodies

from scripts.earth_departure import constants as cst

N = 0

class ApogeeRaising:

	def __init__(self, n_seg, v_inf, tf, tof, ri, Tmax, eps, eps_l, t_last_ap_pass):
		""" Initialization of the `ApogeeRaising` class 

			Parameters
			----------
			n_seg : float
				Number of segments composing the trajectory [-]
			v_inf : float
				S/C excess velocity at Moon encounter [km/s]
			tf : float
				Moon encounter date [MJD2000]
			tof : array
				Time of flight [day]
			ri : array
				S/C initial position and velocity [km] | [km/s]
			Tmax : float
				S/C maximum thrust [kN]
			eps : float
				Ignition arc semi-angle [rad]
			eps_l : float
				Last ignition arc semi-angle [rad]
			l_ap_pass : float
				Date of apogee last pass [day]


		"""

		# Creation of the Sun, Earth and Moon objects
		self.sun = load_bodies.planet('sun')
		self.earth = load_bodies.planet('earth')
		self.moon = load_bodies.planet('moon')

		# Copy of the arguments as attributs
		self.n_seg = n_seg	
		self.v_inf = v_inf
		self.tf = tf 
		self.tof = tof
		self.ri = ri
		self.Tmax = Tmax 
		self.eps = eps
		self.eps_l = eps_l 
		self.t_last_ap_pass = t_last_ap_pass

		# Construction of the grid and time-steps
		self.grid = np.array([i / n_seg for i in range(n_seg + 1)])

		# Computation of the final states
		self.final_states()

		# States and date of last apogee pass
		self.r_dep, self.t_dep = self.get_states_last_apogee_pass()
		self.tof -= self.t_last_ap_pass 

		# Variables boundaries
		self.lb = [0.5 * self.tof] + [self.eps_l / 10.] + [-1, -1, -1] * n_seg
		self.ub = [1.5 * self.tof] + [10. * self.eps_l] + [ 1,  1,  1] * n_seg

	def gradient(self, x):
		return pg.estimate_gradient(lambda x: self.fitness(x), x, 1e-8)

	def get_bounds(self):
		return (self.lb, self.ub)

	def get_nic(self):
		return self.n_seg

	def get_nec(self):
		return 6

	def final_states(self):

		tau = pk.epoch(self.tf, julian_date_type='mjd2000')

		# Computation of the Moon's states at t=tf
		earth_r, earth_v = self.earth.eph(tau)
		earth_r, earth_v = np.array(earth_r), np.array(earth_v)

		# Computation of the Moon's states at t=tf
		moon_r, moon_v = self.moon.eph(tau)
		moon_r, moon_v = np.array(moon_r), np.array(moon_v)

		# Final target position [km] and velocity [km/s]
		self.r_tgt = (moon_r - earth_r) / 1000
		self.v_tgt = (moon_v - earth_v) / 1000 + self.v_inf

		# Computation of the Moon-Earth unitary vector at t=tf
		self.EM_axis = (earth_r - moon_r) / np.linalg.norm(earth_r - moon_r)

	def get_states_last_apogee_pass(self):

		print("Computation of the trajectory before last apogee pass ...")
		t0 = self.tf - tof

		t_span = np.array([t0, t0 + self.t_last_ap_pass]) * pk.DAY2SEC
		t_eval = np.linspace(t_span[0], t_span[-1], 10000)

		solution = solve_ivp(fun=self.R4BP, y0=self.ri, t_span=t_span, t_eval=t_eval, args=(0, 0, 0, self.eps), rtol=1e-12, atol=1e-12)

		return solution.y[:, -1], solution.t[-1]

	def fitness(self, x):

		# Decoding the decision vector 
		tof = x[0]
		eps_l = x[1]
		throttles = np.array([x[2 + 3 * i: 5 + 3 * i] for i in range(self.n_seg)])

		# Equality and inequality constraints vectors
		ceq = list()
		cineq = list()

		# Mismatch and throttles constraints 
		throttle_con = list()
		mismatch_con = list()

		# Throttle constraints
		for t in throttles:
			throttle_con.append(t[0]**2 + t[1]**2 + t[2]**2 - 1)
		cineq.extend(throttle_con)

		# Mismatch constraints
		r_f, v_f = self.propagate(x)
		mismatch_con.extend([a - b for a, b in zip(self.r_tgt, r_f)])
		mismatch_con.extend([a - b for a, b in zip(self.v_tgt, v_f)])
		ceq.extend(mismatch_con)


		# Objective function : minimize the fuel consumption
		obj = sum([np.linalg.norm(throttles[i]) for i in range(len(throttles))], 0) / self.n_seg

		# Assembly of the fitness vector
		retval = [obj]
		retval.extend(ceq)
		retval.extend(cineq)

		global N
		N += 1

		print("({}) Objective : {}".format(N, obj))
		print("     Mismatch constraints : {} km | {} km/s".format(np.linalg.norm(ceq[:3]), np.linalg.norm(ceq[3:])))
		print("     Last arc : {}°".format(eps_l*180/np.pi))
		print("\n")


		return retval


	def propagate(self, x):
		# Decoding the decision vector 
		tof = x[0]
		eps_l = x[1]
		throttles = np.array([x[2 + 3 * i: 5 + 3 * i] for i in range(self.n_seg)])

		# Departure time
		t0 = self.tf - tof 

		# Number of points
		n_points = self.n_seg + 1

		# Return lists
		r = [[0.0] * 3] * n_points  
		v = [[0.0] * 3] * n_points

		# S/C initial position [km] and velocity [km/s]
		ri = self.r_dep[:3]
		vi = self.r_dep[3:]

		# Grid and segment length [day]
		grid = t0 + tof * self.grid

		# Initial conditions
		r[0] = ri
		v[0] = vi 

		# Propagation of the R4BP equations
		for i, tr in enumerate(throttles[:self.n_seg]):

			# Time span [sec]
			t_span = np.array([grid[i], grid[i+1]]) * pk.DAY2SEC
			t_eval = np.linspace(t_span[0], t_span[-1], 100)

			r0 = np.concatenate((r[i], v[i]))

			solution = solve_ivp(fun=self.R4BP, y0=r0, t_span=t_span, t_eval=t_eval, args=(tr[0], tr[1], tr[2], eps_l), rtol=1e-8, atol=1e-8)

			r[i+1] = solution.y[:3, -1]
			v[i+1] = solution.y[3:, -1]

		return r[-1], v[-1]


	def R4BP(self, t, r, tx, ty, tz, eps):

		# Conversion of the time [sec] in days [MJD2000]
		tau = pk.epoch(t * pk.SEC2DAY, julian_date_type='mjd2000')

		# Earth ephemerides [km] | [km/s]
		earth_r, earth_v = self.earth.eph(tau)
		earth_r = np.array(earth_r) / 1000
		earth_v = np.array(earth_v) / 1000

		# Sun and Moon states relative to the Earth [km] | [km/s]
		sun_r, sun_v = self.sun.eph(tau)
		sun_r = np.array(sun_r) / 1000 - earth_r
		sun_v = np.array(sun_v) / 1000 - earth_v

		moon_r, moon_v = self.moon.eph(tau)
		moon_r = np.array(moon_r) / 1000 - earth_r
		moon_v = np.array(moon_v) / 1000 - earth_v

		# S/C states relative to the Earth [km] | [km/s]
		x, y, z, vx, vy, vz = r

		# S/C distances to the Sun, Earth and Moon [km]
		d_E = np.linalg.norm(r[:3])
		d_S = np.linalg.norm(r[:3] - sun_r)
		d_M = np.linalg.norm(r[:3] - moon_r)

		# Sun - Earth distance [km]
		d_ES = np.linalg.norm(sun_r)

		# R4BP equations
		x_dot = vx
		y_dot = vy
		z_dot = vz

		vx_dot = - cst.mu_S / d_S**3 * (x - sun_r[0]) - cst.mu_M / d_M**3 * (x - moon_r[0]) - cst.mu_E / d_E**3 * x + \
					cst.mu_S / d_ES**3 * (-sun_r[0])
		vy_dot = - cst.mu_S / d_S**3 * (y - sun_r[1]) - cst.mu_M / d_M**3 * (y - moon_r[1]) - cst.mu_E / d_E**3 * y + \
					cst.mu_S / d_ES**3 * (-sun_r[1])
		vz_dot = - cst.mu_S / d_S**3 * (z - sun_r[2]) - cst.mu_M / d_M**3 * (z - moon_r[2]) - cst.mu_E / d_E**3 * z + \
					cst.mu_S / d_ES**3 * (-sun_r[2])

		# Thrust arc around perigee
		v = np.linalg.norm(r[3:])
		psi = np.arccos( np.dot(self.EM_axis, r[:3]) / d_E )
		if psi <= eps:
			vx_dot += self.Tmax * vx / v 
			vy_dot += self.Tmax * vy / v 
			vz_dot += self.Tmax * vz / v

		# Correction maneuvers
		vx_dot += self.Tmax * tx
		vy_dot += self.Tmax * ty
		vz_dot += self.Tmax * tz

 
		return [x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot]


if __name__ == '__main__':

	# Kernels loading 
	load_kernels.load()

	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/1', 'rb') as f:
		res = pickle.load(f)

	# Extraction of the S/C excess velocity at Moon departure in the ECLIPJ200 frame [km/s]
	v_inf = np.array([-0.67827715326,  0.63981981778, -0.23054482431])

	# Moon departure date (MJD2000)
	tau = 15128.755128883051

	# Time of flight [days]
	tof = res['t'][-1] * pk.SEC2DAY

	# Initial states [km] | [km/s]
	ri = res['r'][:, 0]

	# S/C maximum thrust [kN]
	Tmax = 1 * 1e-3

	# Ignition angles [rad]
	eps = 2 * np.pi / 180
	eps_l = 0.752151254363917 * np.pi / 180

	# Time of last apogee pass [day]
	t_last_ap_pass = 3056863.02897 * pk.SEC2DAY

	# Definition of the UDP and UDA
	# -----------------------------
	algorithm = load_sqp.load('ipopt')

	n_seg = 100
	udp = ApogeeRaising(n_seg=n_seg, v_inf=v_inf, tf=tau, tof=tof, ri=ri, Tmax=Tmax, eps=eps, eps_l=eps_l, t_last_ap_pass=t_last_ap_pass)
	problem = pg.problem(udp)

	population = pg.population(problem, size=1)


	xi = np.concatenate(([tof - t_last_ap_pass], [eps_l], [0, 0, 0]*n_seg))
	population.set_x(0, xi)

	population = algorithm.evolve(population)






