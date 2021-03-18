#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 2021 10:55:00

@author: SEMBLANET Tom

"""

import matplotlib.pyplot as plt

import pykep as pk 
import pygmo as pg 
import numpy as np 

from scripts.load_bodies import load_asteroid, load_planet
import data.constants as cst

class Earth2Asteroid:
	""" 
	Optimal control problem representing a low-thrust transfer between the Earth environment and a 
	target NEA. The spacecraft leaves the Moon with a infinity velocity `v_inf` relative to the Earth.

	Both the gravitational attraction of the Sun and the Earth are taken into account in this problem, the
	goal is to find the optimal launch date, thrust profil and space path to join the NEA while minimizing 
	the amount of fuel used.

	Parameters:
	-----------
	target: <pykep.planet>
		The target NEA
	n_seg: int
		Number of segment in which the trajectory is divided
	grid_type: string
		Type of grid used, either 'uniform' or 'nonuniform'. If 'nonuniform' is used
		the nodes are denser near the Earth where the dynamic is more sensitive
	t0 : array [<pykep.epoch>, <pykep.epoch>]
		Lower and upper bounds of the launch date
	tof : array [float, float]
		Lower and upper bounds of the time of flight
	m0 : float
		Initial mass of the spacecraft [kg]
	Tmax : float
		Maximum thrust of the spacecraft [N]
	Isp : float
		Specific impulse of the spacecraft [s]

	"""

	def __init__(self, 
				 target = None, 
				 n_seg = 30, 
				 grid_type='uniform', 
				 t0 = [pk.epoch(0), pk.epoch(1000)], 
				 tof = [0, 1000], 
				 m0 = 600, 
				 Tmax = 0.1, 
				 Isp = 2700, 
				 vinf = [0, 2.5e3]):
		""" Initialization of the `Earth2Asteroid` class.

			Parameters:
			-----------
			target: <pykep.planet>
				Target NEO (Near-Earth Object)
			n_seg: int
				Number of segments to use in the problem transcription (time grid)
			grid_type: string
				"uniform" for uniform segments, "nonuniform" to use a denser grid in the first part of the trajectory
			t0: tuple
				List of two pykep.epoch defining the bounds on the launch epoch
			tof: tuple
				List of two floats defining the bounds on the time of flight (days)
			m0: float
				Initial mass of the spacecraft (kg)
			Tmax: float
				Maximum thrust at 1 AU (N)
			Isp: float
				Engine specific impulse (s)
			vinf: array
				Minimal and maximal velocities at infinity relative to the Earth at departure [m/s]
			
		"""

		# Class data members
		self.target = target 
		self.n_seg = n_seg
		self.grid_type = grid_type
		self.sc = pk.sims_flanagan.spacecraft(m0, Tmax, Isp)
		self.earth = load_planet('EARTH')
		self.moon = load_planet('MOON')
		self.vinf = vinf

		# Grid construction
		if grid_type == 'uniform':
			grid = np.array([i / n_seg for i in range(n_seg + 1)])
		elif grid_type == 'nonuniform':
			grid_f = lambda x: x**2 if x < 0.5 else 0.25 + 1.5 * \
				(x - 0.5) # quadratic in [0, 0.5], linear in [0.5, 1]
			grid = np.array([grid_f(i / n_seg) for i in range(n_seg + 1)])

		# Index corresponding to the middle of the transfer
		self.fwd_seg = int(np.searchsorted(grid, 0.5, side='right'))
		self.bwd_seg = n_seg - self.fwd_seg
		
		self.fwd_grid = grid[:self.fwd_seg + 1]
		self.fwd_dt = np.array([(self.fwd_grid[i + 1] - self.fwd_grid[i])
								  for i in range(self.fwd_seg)]) * pk.DAY2SEC

		self.bwd_grid = grid[self.fwd_seg:]
		self.bwd_dt = np.array([(self.bwd_grid[i + 1] - self.bwd_grid[i])
								  for i in range(self.bwd_seg)]) * pk.DAY2SEC

		# Boundaries 
		# [<departure date>, <time of flight>, <final mass>, <vinf_mag>, <vinf_unit>, <throttle[0]>, ..., <throttle[n_seg]>]
		self.lb = [t0[0].mjd2000] + [tof[0]] + [0] + [self.vinf[0]] + [-1, -1, -1] + [-1, -1, -1] * n_seg
		self.ub = [t0[1].mjd2000] + [tof[1]] + [m0] + [self.vinf[1]] + [1, 1, 1] + [1, 1, 1] * n_seg

	def fitness(self, x):
		""" Fitness function of the problem 

			Parameters:
			-----------
			x : array
				Decision vector 

			Returns:
			--------
			retval : array
				Concatenation of the objective function value, equality and inequality constraints

		"""	

		# Objective function : maximize the final mass
		obj = -x[2] / self.sc.mass

		# Equality and inequality constraints vectors
		ceq = list()
		cineq = list()

		throttle_con = list()
		mismatch_con = list()
		vinf_unit_con = list()

		# Throttle constraints : ||T_i|| <= 1
		throttles = [x[7 + 3 * i: 10 + 3 * i] for i in range(self.n_seg)]
		for t in throttles:
			throttle_con.append(t[0]**2 + t[1]**2 + t[2]**2 - 1)
		cineq.extend(throttle_con)

		# Mismatch constraints
		rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, _, _, _, _, dfwd, dbwd = self.propagate(x)
		mismatch_con.extend([a - b for a, b in zip(rfwd[-1], rbwd[0])])
		mismatch_con.extend([a - b for a, b in zip(vfwd[-1], vbwd[0])])
		mismatch_con.extend([mfwd[-1] - mbwd[0]])
		ceq.extend(mismatch_con)

		# Initial velocity : ||vinf_unit|| <= 1
		vinf_unit = x[4:7]
		vinf_unit_con.append(vinf_unit[0]**2 + vinf_unit[1]**2 + vinf_unit[2]**2 - 1)
		ceq.extend(vinf_unit_con)

		# Dimensioning of the mismatch constraints
		ceq[0] /= pk.AU
		ceq[1] /= pk.AU 
		ceq[2] /= pk.AU
		ceq[3] /= pk.EARTH_VELOCITY
		ceq[4] /= pk.EARTH_VELOCITY
		ceq[5] /= pk.EARTH_VELOCITY
		ceq[6] /= self.sc.mass

		# Assembly of the constraint vector
		retval = [obj]
		retval.extend(ceq)
		retval.extend(cineq)

		return retval

	def get_bounds(self):
		""" Returns the lower and upper boundaries of the decision vector 

			Returns:
			--------
			_ : tuple
				Lower and upper boundaries of the decision vector 

		"""
		return (self.lb, self.ub)

	def get_nic(self):
		""" Returns the number of inequality constraints

			Returns:
			--------
			_ : float
				Number of inequality constraints

		"""
		return self.n_seg 

	def get_nec(self):
		""" Returns the number of equality constraints 

			Returns:
			--------
			_ : float
				Number of equality constraints

		"""
		return 8

	def propagate(self, x):
		""" """

		# Decoding of the decision vector
		t0 = x[0]
		tof = x[1]
		mf = x[2]
		vinf_mag = x[3]
		vinf_unit = x[4:7]

		# Extraction of the number of segments for forward and backward propagation
		n_seg = self.n_seg
		fwd_seg = self.fwd_seg
		bwd_seg = self.bwd_seg

		# Extraction of Spacecraft informations
		mi = self.sc.mass
		Tmax = self.sc.thrust
		isp = self.sc.isp 
		veff = isp * pk.G0

		# Extraction of information on the legs
		throttles = [x[7 + 3 * i: 10  + 3 * i] for i in range(n_seg)]

		# Return lists
		n_points_fwd = fwd_seg + 1
		n_points_bwd = bwd_seg + 1

		rfwd = [[0.0] * 3] * (n_points_fwd)
		vfwd = [[0.0] * 3] * (n_points_fwd)
		mfwd = [0.0] * (n_points_fwd)
		ufwd = [[0.0] * 3] * (n_points_fwd)
		dfwd = [[0.0] * 3] * (n_points_fwd)

		rbwd = [[0.0] * 3] * (n_points_bwd) 
		vbwd = [[0.0] * 3] * (n_points_bwd)
		mbwd = [0.0] * (n_points_bwd)
		ubwd = [[0.0] * 3] * (n_points_bwd)
		dbwd = [[0.0] * 3] * (n_points_bwd)

		# Computation of the initial epochs and ephemerides
		ti = pk.epoch(t0)
		ri, vi = self.moon.eph(ti)

		# Adding the initial velocity at infinity (Earth departure)
		vi += vinf_mag * vinf_unit

		# Computation of the final epochs and ephemerides
		tf = pk.epoch(t0 + tof)
		rf, vf = self.target.eph(tf)

		# Forward propagation
		fwd_grid = t0 + tof * self.fwd_grid
		fwd_dt = tof * self.fwd_dt

		# Initial conditions
		rfwd[0] = ri
		vfwd[0] = vi
		mfwd[0] = mi

		# Propagate
		for i, t in enumerate(throttles[:fwd_seg]):
			ufwd[i] = [Tmax * thr for thr in t]

			r_E, v_E = self.earth.eph(pk.epoch(fwd_grid[i]))
			dfwd[i] = [a - b for a, b in zip(r_E, rfwd[i])]
			r3 = sum([r**2 for r in dfwd[i]])**(3 / 2)
			disturbance = [mfwd[i] * cst.MU_EARTH /
						   r3 * ri for ri in dfwd[i]]
			rfwd[i + 1], vfwd[i + 1], mfwd[i + 1] = pk.propagate_taylor_disturbance(
				rfwd[i], vfwd[i], mfwd[i], ufwd[i], disturbance, fwd_dt[i], cst.MU_SUN, veff, -10, -10)

		# Backward propagation
		bwd_grid = t0 + tof * self.bwd_grid
		bwd_dt = tof * self.bwd_dt

		# Final conditions
		rbwd[-1] = rf
		vbwd[-1] = vf
		mbwd[-1] = mf

		# Propagate
		for i, t in enumerate(throttles[-1:-bwd_seg - 1:-1]):
			ubwd[-i - 1] = [Tmax * thr for thr in t]

			r_E, v_E = self.earth.eph(pk.epoch(bwd_grid[-i - 1]))
			dbwd[-i - 1] = [a - b for a, b in zip(r_E, rbwd[-i - 1])]
			r3 = sum([r**2 for r in dbwd[-i - 1]])**(3 / 2)
			disturbance = [mfwd[i] * cst.MU_EARTH /
						   r3 * ri for ri in dbwd[-i - 1]]
			rbwd[-i - 2], vbwd[-i - 2], mbwd[-i - 2] = pk.propagate_taylor_disturbance(
				rbwd[-i - 1], vbwd[-i - 1], mbwd[-i - 1], ubwd[-i - 1], disturbance, -bwd_dt[-i - 1], cst.MU_SUN, veff, -10, -10)

		return rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, ufwd, ubwd, fwd_dt, bwd_dt, dfwd, dbwd

	def gradient(self, x):
		""" Approximates the gradient of the problem.

			Parameters:
			-----------
			dv : array
				Decision vector

			Returns:
			--------
			_ : array
				Gradient of the problem
		"""
		return pg.estimate_gradient(lambda x: self.fitness(x), x, 1e-8)

	def plot_traj(self, x, units=pk.AU, plot_segments=False, plot_thrusts=False, axes=None):
		""" Plots the distance of the spacecraft from the Earth/Sun and the thrust profile

			Parameters:
			-----------
			x: array
				Decision vector returned by the optimizer
			units: float
				Length unit to be used in the plot
			plot_segments: boolean
				If True plots the segments boundaries
			plot_thrusts: boolean
				If True plots the thrust vectors

			Returns:
			--------
			axes: matplotlib.axes
				Axes where to plot

		"""

		import matplotlib as mpl 
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D

		# Creating the axes if necessary
		if axes is None:
			mpl.rcParams['legend.fontsize'] = 10
			fig = plt.figure()
			axes = fig.gca(projection='3d')

		n_seg = self.n_seg
		fwd_seg = self.fwd_seg
		bwd_seg = self.bwd_seg

		t0 = x[0]
		tof = x[1]

		isp = self.sc.isp 
		veff = isp * pk.G0

		fwd_grid = t0 + tof * self.fwd_grid
		bwd_grid = t0 + tof * self.bwd_grid

		throttles = [x[7 + 3 * i: 10 + 3 * i] for i in range(n_seg)]
		alphas = [min(1., np.linalg.norm(t)) for t in throttles]

		times = np.concatenate((fwd_grid, bwd_grid))

		rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, ufwd, ubwd, fwd_dt, bwd_dt, dfwd, dbwd = self.propagate(
			x)

		# Plotting the Sun, the Earth, the Moon and the target
		axes.scatter([0], [0], [0], color='y')
		pk.orbit_plots.plot_planet(self.earth, pk.epoch(
			t0), units=units, legend=True, color=(0.7, 0.7, 1), axes=axes)
		pk.orbit_plots.plot_planet(self.moon, pk.epoch(
			t0), units=units, legend=True, color=(0.7, 0.7, 1), axes=axes)
		pk.orbit_plots.plot_planet(self.target, pk.epoch(
			t0 + tof), units=units, legend=True, color=(0.7, 0.7, 1), axes=axes)

		# Forward propagation
		xfwd = [0.0] * (fwd_seg + 1)
		yfwd = [0.0] * (fwd_seg + 1)
		zfwd = [0.0] * (fwd_seg + 1)
		xfwd[0] = rfwd[0][0] / units
		yfwd[0] = rfwd[0][1] / units
		zfwd[0] = rfwd[0][2] / units

		for i in range(fwd_seg):
			pk.orbit_plots.plot_taylor(rfwd[i], vfwd[i], mfwd[i], ufwd[i], fwd_dt[
										   i], cst.MU_SUN, veff, N=10, units=units, color=(alphas[i], 0, 1 - alphas[i]), axes=axes)

			xfwd[i + 1] = rfwd[i + 1][0] / units
			yfwd[i + 1] = rfwd[i + 1][1] / units
			zfwd[i + 1] = rfwd[i + 1][2] / units

		if plot_segments:
			axes.scatter(xfwd[:-1], yfwd[:-1], zfwd[:-1],
						 label='nodes', marker='o', s=5, c='k')


		# Backward propagation
		xbwd = [0.0] * (bwd_seg + 1)
		ybwd = [0.0] * (bwd_seg + 1)
		zbwd = [0.0] * (bwd_seg + 1)
		xbwd[-1] = rbwd[-1][0] / units
		ybwd[-1] = rbwd[-1][1] / units
		zbwd[-1] = rbwd[-1][2] / units

		for i in range(bwd_seg):
			pk.orbit_plots.plot_taylor(rbwd[-i - 1], vbwd[-i - 1], mbwd[-i - 1], ubwd[-i - 1], -bwd_dt[-i - 1],
										   cst.MU_SUN, veff, N=10, units=units, color=(alphas[-i - 1], 0, 1 - alphas[-i - 1]), axes=axes)
			xbwd[-i - 2] = rbwd[-i - 2][0] / units
			ybwd[-i - 2] = rbwd[-i - 2][1] / units
			zbwd[-i - 2] = rbwd[-i - 2][2] / units
		
		if plot_segments:
			axes.scatter(xbwd[1:], ybwd[1:], zbwd[1:], marker='o', s=5, c='k')

		# Plotting the thrust vectors
		if plot_thrusts:
			throttles = np.array(throttles)
			xlim = axes.get_xlim()
			xrange = xlim[1] - xlim[0]
			ylim = axes.get_ylim()
			yrange = ylim[1] - ylim[0]
			zlim = axes.get_zlim()
			zrange = zlim[1] - zlim[0]

			scale = 0.1

			throttles[:, 0] *= xrange
			throttles[:, 1] *= yrange
			throttles[:, 2] *= zrange

			throttles *= scale

			for (x, y, z, t) in zip(xfwd[:-1] + xbwd[:-1], yfwd[:-1] + ybwd[:-1], zfwd[:-1] + zbwd[:-1], throttles):
				axes.plot([x, x + t[0]], [y, y + t[1]], [z, z + t[2]], c='g')

		return axes

	def plot_dists_thrust(self, x, axes=None):
		""" Plots the distance of the spacecraft from the Earth/Sun and the thrust profile.

			Parameters:
			-----------
			x: array
				Decision vector
			axes: matplotlib.axes
				Axes where to plot

			Returns:
			--------
			axes: matplotlib.axes
				Axes where to plot
		"""

		import matplotlib as mpl
		import matplotlib.pyplot as plt

		# Creating the axes if necessary
		if axes is None:
			mpl.rcParams['legend.fontsize'] = 10
			fig = plt.figure()
			axes = fig.add_subplot(111)

		n_seg = self.n_seg
		fwd_seg = self.fwd_seg
		bwd_seg = self.bwd_seg

		t0 = x[0]
		tof = x[1]

		fwd_grid = t0 + tof * self.fwd_grid
		bwd_grid = t0 + tof * self.bwd_grid

		throttles = [np.linalg.norm(x[7 + 3 * i: 10 + 3 * i])
					 for i in range(n_seg)]

		dist_earth = [0.0] * (n_seg + 2)  # distances spacecraft - Earth
		dist_sun = [0.0] * (n_seg + 2)  # distances spacecraft - Sun
		times = np.concatenate((fwd_grid, bwd_grid))

		rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, ufwd, ubwd, fwd_dt, bwd_dt, _, _ = self.propagate(
			x)

		# Forward propagation
		xfwd = [0.0] * (fwd_seg + 1)
		yfwd = [0.0] * (fwd_seg + 1)
		zfwd = [0.0] * (fwd_seg + 1)
		xfwd[0] = rfwd[0][0] / pk.AU
		yfwd[0] = rfwd[0][1] / pk.AU
		zfwd[0] = rfwd[0][2] / pk.AU

		r_E = [ri / pk.AU for ri in self.earth.eph(pk.epoch(fwd_grid[0]))[0]]
		dist_earth[0] = np.linalg.norm(
			[r_E[0] - xfwd[0], r_E[1] - yfwd[0], r_E[2] - zfwd[0]])
		dist_sun[0] = np.linalg.norm([xfwd[0], yfwd[0], zfwd[0]])

		for i in range(fwd_seg):
			xfwd[i + 1] = rfwd[i + 1][0] / pk.AU
			yfwd[i + 1] = rfwd[i + 1][1] / pk.AU
			zfwd[i + 1] = rfwd[i + 1][2] / pk.AU
			r_E = [
				ri / pk.AU for ri in self.earth.eph(pk.epoch(fwd_grid[i + 1]))[0]]
			dist_earth[
				i + 1] = np.linalg.norm([r_E[0] - xfwd[i + 1], r_E[1] - yfwd[i + 1], r_E[2] - zfwd[i + 1]])
			dist_sun[
				i + 1] = np.linalg.norm([xfwd[i + 1], yfwd[i + 1], zfwd[i + 1]])

		# Backward propagation
		xbwd = [0.0] * (bwd_seg + 1)
		ybwd = [0.0] * (bwd_seg + 1)
		zbwd = [0.0] * (bwd_seg + 1)
		xbwd[-1] = rbwd[-1][0] / pk.AU
		ybwd[-1] = rbwd[-1][1] / pk.AU
		zbwd[-1] = rbwd[-1][2] / pk.AU

		r_E = [
			ri / pk.AU for ri in self.earth.eph(pk.epoch(bwd_grid[-1]))[0]]
		dist_earth[-1] = np.linalg.norm([r_E[0] - xbwd[-1],
										 r_E[1] - ybwd[-1], r_E[2] - zbwd[-1]])
		dist_sun[-1] = np.linalg.norm([xbwd[-1], ybwd[-1], zbwd[-1]])

		for i in range(bwd_seg):
			xbwd[-i - 2] = rbwd[-i - 2][0] / pk.AU
			ybwd[-i - 2] = rbwd[-i - 2][1] / pk.AU
			zbwd[-i - 2] = rbwd[-i - 2][2] / pk.AU
			r_E = [
				ri / pk.AU for ri in self.earth.eph(pk.epoch(bwd_grid[-i - 2]))[0]]
			dist_earth[-i - 2] = np.linalg.norm(
				[r_E[0] - xbwd[-i - 2], r_E[1] - ybwd[-i - 2], r_E[2] - zbwd[-i - 2]])
			dist_sun[-i -
					 2] = np.linalg.norm([xbwd[-i - 2], ybwd[-i - 2], zbwd[-i - 2]])

		axes.set_xlabel("t [mjd2000]")
		# Plot Earth distance
		axes.plot(times, dist_earth, c='b', label="sc-Earth")
		# Plot Sun distance
		axes.plot(times, dist_sun, c='y', label="sc-Sun")
		axes.set_ylabel("distance [AU]", color='k')
		axes.set_ylim(bottom=0.)
		axes.tick_params('y', colors='k')
		axes.legend(loc=2)

		# Plot thrust profile
		axes = axes.twinx()
			
		thrusts = throttles.copy()
		# duplicate the last for plotting
		thrusts = np.append(thrusts, thrusts[-1])
		axes.step(np.concatenate(
			(fwd_grid, bwd_grid[1:])), thrusts, where="post", c='r', linestyle='--')
		axes.set_ylabel("T/Tmax$_{1AU}$", color='r')
		axes.tick_params('y', colors='r')
		axes.set_xlim([times[0], times[-1]])
		axes.set_ylim([0, max(thrusts) + 0.2])

		return axes

	def get_name(self):
		return "Low-Thrust transfer between Earth and NEOs - Preliminary design (Free Velocity at Infinity + Departure from Moon)"

	def get_extra_info(self):
		retval = "\tTarget NEO: " + self.target.name
		retval += "\n\tStart mass: " + str(self.sc.mass) + " kg"
		retval += "\n\tMaximum thrust as 1AU: " + str(self.sc.thrust) + " N"
		retval += "\n\tSpecific impulse: " + str(self.sc.isp) + " s"
		retval += "\n\n\tLaunch window: [" + \
			str(self.lb[0]) + ", " + str(self.ub[0]) + "] - MJD2000"
		retval += "\n\tBounds on time of flight: [" + str(
			self.lb[1]) + ", " + str(self.ub[1]) + "] - days"
		retval += "\n\n\tNumber of segments: " + str(self.n_seg)
		retval += "\n\tGrid type: " + self.grid_type

		return retval

	def report(self, x):
		"""
		Prints human readable information on the trajectory represented by the decision vector x

		Parameters:
		-----------
		x : array
			Decision vector
	   
		"""

		# Decoding the decision vector
		n_seg = self.n_seg
		mi = self.sc.mass
		t0 = x[0]
		tof = x[1]
		mf = x[2]
		vinf_mag = x[3]
		vinf_unit = x[4:7]
		thrusts = [np.linalg.norm(x[7 + 3 * i: 10 + 3 * i])
				   for i in range(n_seg)]

		tf = t0 + tof
		mP = mi - mf
		deltaV = self.sc.isp * pk.G0 * np.log(mi / mf)

		dt = np.append(self.fwd_dt, self.bwd_dt) * tof / pk.DAY2SEC
		time_thrusts_on = sum(dt[i] for i in range(
			len(thrusts)) if thrusts[i] > 0.1)

		vinf_dep = vinf_mag*vinf_unit

		print("Departure:", pk.epoch(t0), "(", t0, "mjd2000)")
		print("Time of flight:", tof, "days")
		print("Arrival:", pk.epoch(tf), "(", tf, "mjd2000)")
		print("Delta-v:", deltaV, "m/s")
		print("Propellant consumption:", mP, "kg")
		print("Thrust-on time:", time_thrusts_on, "days")
		print("Initial velocity at infinity vector: {}".format(vinf_dep))
		print("Initial velocity at infinity magnitude: {} km/s".format(np.linalg.norm(vinf_dep) / 1000))


