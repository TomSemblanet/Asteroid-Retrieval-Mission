import matplotlib.pyplot as plt

import sys

import pykep as pk 
import pygmo as pg 
import numpy as np

from scripts.utils import load_sqp, load_kernels, load_bodies

from data import constants as cst

""" NEA - Earth Trajectory with fixed departure """

class NEA2Earth_Pos:

	def __init__(self, nea, n_seg, t0, tof, m0, Tmax, Isp, nea_mass, phi_min, phi_max, theta_min, theta_max, earth_grv=True):

		# Creation of the planet and NEA objects
		self.nea = nea
		self.earth = load_bodies.planet('earth')
		self.moon = load_bodies.planet('moon')

		# Creation of the spacecraft object
		self.sc = pk.sims_flanagan.spacecraft(m0, Tmax, Isp)

		# Copy of the arguments as attributs
		self.n_seg = n_seg
		self.t0 = t0
		self.tof = tof
		self.earth_grv = earth_grv
		self.phi_min = phi_min
		self.phi_max = phi_max
		self.theta_min = theta_min
		self.theta_max = theta_max
		self.nea_mass = nea_mass

		# Some constants
		self.earth_SOI = 0.925e9

		# Grid construction
		grid_f = lambda x: x**2 if x < 0.5 else 0.25 + 1.5 * (x - 0.5) 
		grid = np.flip(np.array([1 - grid_f(i / n_seg) for i in range(n_seg + 1)]))

		# Number of forward (fwd) and backward (bwd) segments
		self.n_fwd_seg = int(np.searchsorted(grid, 0.5, side='right'))
		self.n_bwd_seg = n_seg - self.n_fwd_seg

		# Index of forward and backward segments
		self.fwd_grid = grid[:self.n_fwd_seg + 1]
		self.bwd_grid = grid[self.n_fwd_seg:]

		# Time-step for each segments
		self.fwd_dt = (self.fwd_grid[1:] - self.fwd_grid[:-1]) * pk.DAY2SEC
		self.bwd_dt = (self.bwd_grid[1:] - self.bwd_grid[:-1]) * pk.DAY2SEC

		# Boundaries [<departure_date>, <time of flight>, <final mass>, <phi>, <theta>, <throttle[0]>, ..., <throttle[n_seg-1]>]
		self.lb = [t0[0].mjd2000] + [tof[0]] + [nea_mass] + [phi_min] + [theta_min] + [-1, -1, -1] * n_seg
		self.ub = [t0[1].mjd2000] + [tof[1]] + [m0] + [phi_max] + [theta_max] + [1, 1, 1] * n_seg

	def gradient(self, x):
		return pg.estimate_gradient(lambda x: self.fitness(x), x, 1e-8)

	def get_bounds(self):
		return (self.lb, self.ub)

	def get_nic(self):
		return self.n_seg 

	def get_nec(self):
		return 0

	def fitness(self, x):

		# Decoding of the decision vector
		t0 = x[0]
		tof = x[1]
		mf = x[2]
		phi = x[3]
		theta = x[4]
		throttles = np.array([x[5 + 3 * i: 8 + 3 * i] for i in range(self.n_seg)])

		# Propagation of the dynamics equations
		rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, _, _, _, _, _, _ = self.propagate(x)

		# Objective function : maximization of the final mass
		obj = np.linalg.norm([a - b for a, b in zip(rfwd[-1], rbwd[0])]) / pk.AU

		# Inequality constraints vectors
		cineq = list()

		# Throttle, mismatch and arrival position constraints vectors
		throttle_con = list()

		# Throttle constraints
		for t in throttles:
			throttle_con.append(t[0]**2 + t[1]**2 + t[2]**2 - 1)
		cineq.extend(throttle_con)

		# Assembly of the fitness vector
		retval = [obj]
		retval.extend(cineq)

		return retval

	def propagate(self, x):

		# Extraction of the number of segments for forward and backward propagation
		n_seg = self.n_seg
		n_fwd_seg = self.n_fwd_seg
		n_bwd_seg = self.n_bwd_seg

		# Decoding the decision vector
		t0 = x[0]
		tof = x[1]
		mf  = x[2]
		phi = x[3]
		theta = x[4]
		throttles = np.array([x[5 + 3 * i: 8 + 3 * i] for i in range(n_seg)])

		# Extraction of the spacecraft informations
		m0 = self.sc.mass
		Tmax = self.sc.thrust
		isp = self.sc.isp
		veff = self.sc.isp * pk.G0

		# Number of forward and backward points
		n_points_fwd = n_fwd_seg + 1
		n_points_bwd = n_bwd_seg + 1

		# Return lists
		rfwd = [[0.0] * 3] * n_points_fwd  # Positions array
		vfwd = [[0.0] * 3] * n_points_fwd  # Velocities array
		mfwd = [0.0] * n_points_fwd		# Masses array
		ufwd = [[0.0] * 3] * n_points_fwd  # Unit throttles array
		dfwd = [[0.0] * 3] * n_points_fwd  # Distance Spacecraft / Earth

		rbwd = [[0.0] * 3] * n_points_bwd  # Positions array
		vbwd = [[0.0] * 3] * n_points_bwd  # Velocities array
		mbwd = [0.0] * n_points_bwd		# Masses array
		ubwd = [[0.0] * 3] * n_points_bwd  # Unit throttles array
		dbwd = [[0.0] * 3] * n_points_bwd  # Distance Spacecraft / Earth

		# Computation of the initial ephemerides (Departure from the NEA)
		ti = pk.epoch(t0)
		ri, vi = self.nea.eph(ti)

		# Computation of the final ephemerides (Arrival at the Earth)
		tf = pk.epoch(t0 + tof)
		r_e, v_e = self.earth.eph(tf)

		# --------------------------------------------------------

		# Coordinate system
		i_unt = (v_e / np.linalg.norm(v_e))
		k_unt = np.cross(r_e / np.linalg.norm(r_e), i_unt)
		j_unt = np.cross(k_unt, i_unt)

		# Spherical coordinates
		rho = self.earth_SOI

		# Cartesian coordinates
		i = rho * np.cos(phi) * np.sin(theta)
		j = rho * np.sin(phi) * np.sin(theta)
		k = rho * np.cos(theta)

		# Construction of the vector in the new base
		arr_vec = i * i_unt + j * j_unt + k * k_unt

		# # ==============================================================================

		# print("Phi : {}°".format(phi * cst.RAD2DEG))
		# print("Theta : {}°".format(theta * cst.RAD2DEG))

		# phi_deg = phi * cst.RAD2DEG
		# theta_deg = theta * cst.RAD2DEG

		# fig = plt.figure()
		# ax = fig.gca(projection='3d')

		# # Plot the Earth orbit
		# pk.orbit_plots.plot_planet(self.earth, pk.epoch(t0 + tof), units=pk.AU, legend=True, color=(0.7, 0.7, 1), axes=ax)

		# # Plot the Earth local coordinate frame
		# ax.plot([r_e[0]/pk.AU, r_e[0]/pk.AU+ i_unt[0]], [r_e[1]/pk.AU, r_e[1]/pk.AU+ i_unt[1]], [r_e[2]/pk.AU, r_e[2]/pk.AU+ i_unt[2]], label='i')
		# ax.plot([r_e[0]/pk.AU, r_e[0]/pk.AU+ j_unt[0]], [r_e[1]/pk.AU, r_e[1]/pk.AU+ j_unt[1]], [r_e[2]/pk.AU, r_e[2]/pk.AU+ j_unt[2]], label='j')
		# ax.plot([r_e[0]/pk.AU, r_e[0]/pk.AU+ k_unt[0]], [r_e[1]/pk.AU, r_e[1]/pk.AU+ k_unt[1]], [r_e[2]/pk.AU, r_e[2]/pk.AU+ k_unt[2]], label='k')

		# # Plot the arrival point
		# ax.plot(np.array([r_e[0] + arr_vec[0]]) / pk.AU, np.array([r_e[1] + arr_vec[1]]) / pk.AU, np.array([r_e[2] + arr_vec[2]]) / pk.AU, marker='x')
		
		# # Plot the Sun
		# ax.plot([0], [0], [0], marker='o', color='yellow')

		# ax.set_xlim(r_e[0] / pk.AU - 0.01, r_e[0] / pk.AU + 0.01)
		# ax.set_ylim(r_e[1] / pk.AU - 0.01, r_e[1] / pk.AU + 0.01)
		# ax.set_zlim(r_e[2] / pk.AU - 0.01, r_e[2] / pk.AU + 0.01)

		# plt.legend()
		# plt.show()

		# import sys
		# sys.exit()

		# # ==============================================================================

		rf = r_e + arr_vec
		vf = - arr_vec / np.linalg.norm(arr_vec) * np.linalg.norm(v_e)

		# Forward propagation
		# -------------------

		# Forward grid and segment length
		fwd_grid = t0 + tof * self.fwd_grid
		fwd_dt = tof * self.fwd_dt

		# Initial conditions
		rfwd[0] = ri
		vfwd[0] = vi 
		mfwd[0] = m0

		# Propagate
		for i, t in enumerate(throttles[:n_fwd_seg]):
			ufwd[i] = [Tmax * thr for thr in t]

			if self.earth_grv == True:
				# Earth gravity disturbance
				r_E, v_E = self.earth.eph(pk.epoch(fwd_grid[i]))
				dfwd[i] = [a - b for a, b in zip(r_E, rfwd[i])]
				r3 = sum([r**2 for r in dfwd[i]])**(3 / 2)

				disturbance = [mfwd[i] * pk.MU_EARTH / r3 * ri for ri in dfwd[i]]

				rfwd[i + 1], vfwd[i + 1], mfwd[i + 1] = pk.propagate_taylor_disturbance(
	   				  rfwd[i], vfwd[i], mfwd[i], ufwd[i], disturbance, fwd_dt[i], pk.MU_SUN, veff, -10, -10)

				if i == len(rfwd) - 2:
					r_E_next, _ = self.earth.eph(pk.epoch(fwd_grid[i + 1]))
					dfwd[i + 1] = [a - b for a, b in zip(r_E_next, rfwd[i + 1])]
			
			else:
				rfwd[i + 1], vfwd[i + 1], mfwd[i + 1] = pk.propagate_taylor(
	   				  rfwd[i], vfwd[i], mfwd[i], ufwd[i], fwd_dt[i], pk.MU_SUN, veff, -10, -10)

		# Backaward propagation
		# ---------------------

		# Backward grid and segment length
		bwd_grid = t0 + tof * self.bwd_grid
		bwd_dt = tof * self.bwd_dt

		# Final conditions
		rbwd[-1] = rf
		vbwd[-1] = vf 
		mbwd[-1] = mf

		# Propagate
		for i, t in enumerate(throttles[-1:-n_bwd_seg - 1: -1]):
			ubwd[-1 - i] = [Tmax * thr for thr in t]

			if self.earth_grv == True:
				# Earth gravity disturbance
				r_E, v_E = self.earth.eph(pk.epoch(bwd_grid[-1 - i]))
				dbwd[-1 - i] = [a - b for a, b in zip(r_E, rbwd[-1 - i])]
				r3 = sum([r**2 for r in dbwd[-1 - i]])**(3 / 2)

				disturbance = [mbwd[-1 - i] * pk.MU_EARTH / r3 * ri for ri in dbwd[-1 - i]]

				rbwd[-1 - (i+1)], vbwd[-1 - (i+1)], mbwd[-1 - (i+1)] = pk.propagate_taylor_disturbance(
					  rbwd[-1 - i], vbwd[-1 - i], mbwd[-1 - i], ubwd[-1 - i], disturbance, -bwd_dt[-1 - i], pk.MU_SUN, veff, -10, -10)

				if (len(dbwd) - 1) - (i + 1) == 0:
					r_E, v_E = self.earth.eph(pk.epoch(bwd_grid[-1 - (i+1)]))
					dbwd[-1 - (i+1)] = [a - b for a, b in zip(r_E, rbwd[-1 - (i+1)])]

			else:
				rbwd[-1 - (i+1)], vbwd[-1 - (i+1)], mbwd[-1 - (i+1)] = pk.propagate_taylor(
					  rbwd[-1 - i], vbwd[-1 - i], mbwd[-1 - i], ubwd[-1 - i], -bwd_dt[-1 - i], pk.MU_SUN, veff, -10, -10)

		return rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, ufwd, ubwd, fwd_dt, bwd_dt, dfwd, dbwd

	def get_deltaV(self, x):

		# Propagation
		_, _, _, _, mfwd, mbwd, _, _, _, _, _, _ = self.propagate(x)

		# Get the mass error [kg]
		mass_err = (mfwd[-1] - mbwd[0])

		# Final mass [kg]
		mf = x[2]
		
		# Initial and final mass [kg]
		mi = self.sc.mass

		# Correction of the final mass [kg]
		mf_corr = mf + mass_err

		deltaV = self.sc.isp * cst.G0 * np.log(mi / mf_corr)
		
		return deltaV

	def plot_traj(self, x):

		fig = plt.figure()
		ax = fig.gca(projection='3d')

		# Extraction of the number of segment of forward and backward propagation
		n_seg = self.n_seg 
		n_fwd_seg = self.n_fwd_seg 
		n_bwd_seg = self.n_bwd_seg

		# Time of flight
		t0 = x[0]
		tof = x[1]

		# Spacecraft characteristic
		isp = self.sc.isp
		veff = isp * pk.G0

		# Reconstruction of the forward and backward grid
		fwd_grid = t0 + tof * self.fwd_grid
		bwd_grid = t0 + tof * self.bwd_grid

		# Thrust
		throttles = [x[5 + 3 * i : 8 + 3 * i] for i in range(n_seg)]
		alphas = [min(1., np.linalg.norm(t)) for t in throttles]

		# Time vector
		times = np.concatenate((fwd_grid, bwd_grid))

		# Propagation 
		rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, ufwd, ubwd, fwd_dt, bwd_dt, dfwd, dbwd = self.propagate(x)

		# Plotting the Sun, Earth and NEA
		ax.plot([0], [0], [0], color='y')
		pk.orbit_plots.plot_planet(self.earth, pk.epoch(
			t0 + tof), units=pk.AU, color=(0.7, 0.7, 1), axes=ax)
		pk.orbit_plots.plot_planet(self.nea, pk.epoch(
			t0), units=pk.AU, legend=True, color=(0.7, 0.7, 1), axes=ax)

		# Forward propagation for plotting
		xfwd = [0.0] * (n_fwd_seg + 1)
		yfwd = [0.0] * (n_fwd_seg + 1)
		zfwd = [0.0] * (n_fwd_seg + 1)

		xfwd[0] = rfwd[0][0] / pk.AU
		yfwd[0] = rfwd[0][1] / pk.AU
		zfwd[0] = rfwd[0][2] / pk.AU

		for i in range(n_fwd_seg):

			if self.earth_grv == True:
				r3 = sum([r**2 for r in dfwd[i]])**(3 / 2)
				disturbance = [mfwd[i] * pk.MU_EARTH / r3 * ri for ri in dfwd[i]]

				pk.orbit_plots.plot_taylor_disturbance(rfwd[i], vfwd[i], mfwd[i], ufwd[i], disturbance, fwd_dt[
											   i], cst.MU_SUN, veff, N=100, units=pk.AU, color=(alphas[i], 0, 1 - alphas[i]), axes=ax)

			else:
				pk.orbit_plots.plot_taylor(rfwd[i], vfwd[i], mfwd[i], ufwd[i], fwd_dt[
											   i], cst.MU_SUN, veff, N=100, units=pk.AU, color=(alphas[i], 0, 1 - alphas[i]), axes=ax)

			xfwd[i + 1] = rfwd[i + 1][0] / pk.AU
			yfwd[i + 1] = rfwd[i + 1][1] / pk.AU
			zfwd[i + 1] = rfwd[i + 1][2] / pk.AU

		# Backward propagation for plotting
		xbwd = [0.0] * (n_bwd_seg + 1)
		ybwd = [0.0] * (n_bwd_seg + 1)
		zbwd = [0.0] * (n_bwd_seg + 1)

		xfwd[-1] = rbwd[-1][0] / pk.AU
		yfwd[-1] = rbwd[-1][1] / pk.AU
		zfwd[-1] = rbwd[-1][2] / pk.AU

		for i in range(n_bwd_seg):

			if self.earth_grv == True:
				r3 = sum([r**2 for r in dbwd[-1 - i]])**(3 / 2)
				disturbance = [mbwd[i] * pk.MU_EARTH / r3 * ri for ri in dbwd[-1 - i]]

				pk.orbit_plots.plot_taylor_disturbance(rbwd[-i - 1], vbwd[-i - 1], mbwd[-i - 1], ubwd[-i - 1], disturbance, -bwd_dt[-i - 1],
											   cst.MU_SUN, veff, N=100, units=pk.AU, color=(alphas[-i - 1], 0, 1 - alphas[-i - 1]), axes=ax)

			else:
				pk.orbit_plots.plot_taylor(rbwd[-i - 1], vbwd[-i - 1], mbwd[-i - 1], ubwd[-i - 1], -bwd_dt[-i - 1],
											   cst.MU_SUN, veff, N=100, units=pk.AU, color=(alphas[-i - 1], 0, 1 - alphas[-i - 1]), axes=ax)

			xbwd[-1 - (i + 1)] = rbwd[-1 - (i + 1)][0] / pk.AU
			ybwd[-1 - (i + 1)] = rbwd[-1 - (i + 1)][1] / pk.AU
			zbwd[-1 - (i + 1)] = rbwd[-1 - (i + 1)][2] / pk.AU

		ax.scatter(xfwd[:-1], yfwd[:-1], zfwd[:-1], marker='o', s=5, c='k')
		ax.scatter(xbwd[1:], ybwd[1:], zbwd[1:], marker='o', s=5, c='k')

		r_e, v_e = self.earth.eph(t0 + tof)

		return fig, ax

	def plot_thrust(self, x):

		fig = plt.figure()
		ax = fig.add_subplot(111)

		# Extraction of the number of segment of forward and backward propagation
		n_seg = self.n_seg 
		n_fwd_seg = self.n_fwd_seg 
		n_bwd_seg = self.n_bwd_seg 

		# Time of flight
		t0 = x[0]
		tof = x[1]

		# Reconstruction of the forward and backward grid
		fwd_grid = t0 + tof * self.fwd_grid
		bwd_grid = t0 + tof * self.bwd_grid

		# Thrust
		throttles = [x[5 + 3 * i : 8 + 3 * i] for i in range(n_seg)]

		# Time vector
		times = np.concatenate((fwd_grid[:], bwd_grid[1:-1]))
		throttles_mg = [self.sc.thrust * np.linalg.norm(t) for t in throttles]

		ax.plot(times, throttles_mg)

		ax.set_xlabel('Day (mjd2000)')
		ax.set_ylabel('Thrust (N)')

		plt.grid()
		
		return fig, ax

	def report(self, x, print_=True):

		# Decoding the decision vector
		n_seg = self.n_seg
		mi = self.sc.mass
		t0 = x[0]
		tof = x[1]
		mf = x[2]
		phi = x[3]
		theta = x[4]
		thrusts = [np.linalg.norm(x[5 + 3 * i: 8 + 3 * i])
				   for i in range(n_seg)]

		tf = t0 + tof
		mP = mi - mf
		deltaV = self.sc.isp * cst.G0 * np.log(mi / mf)

		dt = np.append(self.fwd_dt, self.bwd_dt) * tof / cst.DAY2SEC
		time_thrusts_on = sum(dt[i] for i in range(
			len(thrusts)) if thrusts[i] > 0.1)

		_, v_e = self.earth.eph(tf)
		_, _, _, vbwd, _, _, _, _, _, _, _, _ = self.propagate(x) 


		if print_ == True:
			print("Departure:", pk.epoch(t0), "(", t0, "mjd2000)")
			print("Time of flight:", tof, "days")
			print("Arrival:", pk.epoch(tf), "(", tf, "mjd2000)")
			print("Delta-v:", deltaV, "m/s")
			print("Propellant consumption:", mP, "kg")
			print("Thrust-on time:", time_thrusts_on, "days")
			print("Phi:", phi * cst.RAD2DEG)
			print("Theta:", theta * cst.RAD2DEG)
			print("Earth arrival dV: {} km/s".format(np.linalg.norm(v_e - vbwd[-1]) / 1000))

		else:
			return '\n'.join(["Departure:" + str(pk.epoch(t0)) + "(" + str(t0) + "mjd2000)", 
							  "Time of flight:" + str(tof) + "days",
							  "Arrival:" + str(pk.epoch(tf)), "(" + str(tf) + "mjd2000)",
							  "Delta-v:"+ str(deltaV) + "m/s",
							  "Propellant consumption:"+ str(mP) + "kg",
							  "Thrust-on time:"+  str(time_thrusts_on) +  "days",
							  "Phi:" + str(phi * cst.RAD2DEG),
							  "Theta:" + str(theta * cst.RAD2DEG),
							  "Earth arrival dV: "+ str(np.linalg.norm(v_e - vbwd[-1]) / 1000) + " km/s"])


	def check_con_violation(self, x, print_=True):

		fitness_vec = self.fitness(x)

		obj = fitness_vec[0]
		ceq = fitness_vec[1:8]
		cineq = fitness_vec[8:]

		thrust_bool = True
		for i in range(3, len(x)):
			if (x[i] >= self.lb[i] and x[i] <= self.ub[i]):
				thrust_bool = True
			else:
				thrust_bool = False

		if print_ == True:
			print("Variables:\n-----------\n")
			print("Departure date :\n\t {}\n".format( x[0] >= self.lb[0] and x[0] <= self.ub[0] ))
			print("Time of flight :\n\t {}\n".format( x[1] >= self.lb[1] and x[1] <= self.ub[1] ))
			print("Final mass :\n\t {}\n".format( x[2] >= self.lb[2] and x[2] <= self.ub[2] ))
			print("Thrust :\n\t {}\n".format(thrust_bool))	

			print("Equality constraints:\n----------------------\n")
			print("dX : {} km".format(ceq[0] * pk.AU / 1000))
			print("dY : {} km".format(ceq[1] * pk.AU / 1000))
			print("dZ : {} km".format(ceq[2] * pk.AU / 1000))
			print("dVX : {} km/s".format(ceq[3] * pk.EARTH_VELOCITY / 1000))
			print("dVY : {} km/s".format(ceq[4] * pk.EARTH_VELOCITY / 1000))
			print("dVZ : {} km/s".format(ceq[5] * pk.EARTH_VELOCITY / 1000))
			print("dM : {} kg".format(ceq[6] * self.sc.mass))

			print("Inequality constraints:\n------------------------\n")
			print("Thrust :\n")
			for i, cineq_ in enumerate(cineq):
				print("<{}> : {}\t{}".format(i, True if cineq_<=0 else cineq_, cineq_+1))
			print("\n\n")

		else:
			return '\n'.join(["Variables:\n-----------\n",
							  "Departure date :\n\t {}\n".format( x[0] >= self.lb[0] and x[0] <= self.ub[0] ),
							  "Time of flight :\n\t {}\n".format( x[1] >= self.lb[1] and x[1] <= self.ub[1] ),
							  "Final mass :\n\t {}\n".format( x[2] >= self.lb[2] and x[2] <= self.ub[2] ),
							  "Thrust :\n\t {}\n".format(thrust_bool),
							  "Equality constraints:\n----------------------\n",
							  "dX : {} km".format(ceq[0] * pk.AU / 1000),
							  "dY : {} km".format(ceq[1] * pk.AU / 1000), 
							  "dZ : {} km".format(ceq[2] * pk.AU / 1000),
							  "dVX : {} km/s".format(ceq[3] * pk.EARTH_VELOCITY / 1000),
							  "dVY : {} km/s".format(ceq[4] * pk.EARTH_VELOCITY / 1000),
							  "dVZ : {} km/s".format(ceq[5] * pk.EARTH_VELOCITY / 1000),
							  "dM : {} kg".format(ceq[6] * self.sc.mass),
							  "Inequality constraints:\n------------------------\n",
							  "Thrust :\n"] + 
							  ["<{}> : {}\t{}".format(i, True if cineq_<=0 else cineq_, cineq_+1) for i, cineq_ in enumerate(cineq)])


	def brief(self, x):

		# Decoding the decision vector
		n_seg = self.n_seg
		mi = self.sc.mass
		t0 = x[0]
		tof = x[1]
		mf = x[2]
		thrusts = [np.linalg.norm(x[3 + 3 * i: 6 + 3 * i])
				   for i in range(n_seg)]

		tf = t0 + tof
		mP = mi - mf
		deltaV = self.get_deltaV(x)

		dt = np.append(self.fwd_dt, self.bwd_dt) * tof / cst.DAY2SEC
		time_thrusts_on = sum(dt[i] for i in range(
			len(thrusts)) if thrusts[i] > 0.1)

		fitness_vec = self.fitness(x)
		obj = fitness_vec[0] * pk.AU / 1000

		print("Departure:", pk.epoch(t0), "(", t0, "mjd2000)")
		print("Time of flight:", tof, "days")
		print("Arrival:", pk.epoch(tf), "(", tf, "mjd2000)")
		print("Delta-v:", deltaV, "m/s")
		print("Propellant consumption:", mP, "kg")
		print("Thrust-on time:", time_thrusts_on, "days")

		print("Position error : {} km".format(obj))