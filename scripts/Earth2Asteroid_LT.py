#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 2021 11:00:23

@author: SEMBLANET Tom


- Infinity velocity at Earth departure : fixed at 0 km/s
- Earth's gravity not taken in account

"""

import matplotlib.pyplot as plt

import pykep as pk 
import pygmo as pg 
import numpy as np 
import math as mt

from scripts.load_bodies import load_asteroid, load_planet

# Some constants
MU_SUN = 132712440018e9 # [m^3/s^2]
MU_EARTH = 398600.4418e9 # [m^3/s^2]
R_EARTH = 6371.0084e3 # [m]

class Earth2Asteroid:
	""" 
	This class is a User-Defined Problem (UDP) which can be used with the Pygmo
	open-source library. It represents a low-thrust trajectory from the Earth to
	a target NEO. The trajectory is modeled using the Sims-Flanagan model, extended to include the
	Earth's gravity (assumed constant along each segment).
	The propulsion model can be both nuclear (NEP) or solar (SEP).

	"""

	def __init__(self, target, n_seg, grid_type, t0, tof, m0, Tmax, Isp, sep):
		""" Initialization of the `Earth2Asteroid` class.

			Parameters:
			-----------
			target: <pykep.planet>
				Target NEO (Near-Earth Object)
			n_seg : int
				Number of segments to use in the problem transcription (time grid)
			grid_type : string
				"uniform" for uniform segments, "nonuniform" to use a denser grid in the first part of the trajectory
			t0 : tuple
				List of two pykep.epoch defining the bounds on the launch epoch
			tof : tuple
				List of two floats defining the bounds on the time of flight (days)
			m0 : float
				Initial mass of the spacecraft (kg)
			Tmax : float
				Maximum thrust at 1 AU (N)
			Isp : float
				Engine specific impulse (s)
			sep : boolean
				Activates a Solar Electric Propulsion model for the thrust - distance dependency.
			
		"""

		# Class data members
		self.target = target 
		self.n_seg = n_seg
		self.grid_type = grid_type
		self.sc = pk.sims_flanagan.spacecraft(m0, Tmax, Isp)
		self.earth = load_planet('EARTH')
		self.sep = sep 

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
		# [<departure date>, <time of flight>, <final mass>, <throttle[0]>, ..., <throttle[n_seg]>]
		self.lb = [t0[0].mjd2000] + [tof[0]] + [0] + [-1, -1, -1] * n_seg
		self.ub = [t0[1].mjd2000] + [tof[1]] + [m0] + [1, 1, 1] * n_seg

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
		obj = -x[2]

		# Constraints 
		ceq = list()
		cineq = list()
		throttle_con = list()
		mismatch_con = list()

		# Throttle constraints : ||T_i|| <= 1
		throttles = [x[3 + 3 * i: 6 + 3 * i] for i in range(self.n_seg)]
		for t in throttles:
			throttle_con.append(t[0]**2 + t[1]**2 + t[2]**2 - 1)
		cineq.extend(throttle_con)

		# Mismatch constraints
		rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, _, _, _, _, dfwd, dbwd = self.propagate(x)
		mismatch_con.extend([a - b for a, b in zip(rfwd[-1], rbwd[0])])
		mismatch_con.extend([a - b for a, b in zip(vfwd[-1], vbwd[0])])
		mismatch_con.extend([mfwd[-1] - mbwd[0]])
		ceq.extend(mismatch_con)

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
		return 7

	def sep_model(self, r):
		""" Implementation of the Solar Electric Propulsion w.r.t the distance to
			the Sun

			Parameters:
			-----------
			r : array
				Spacecraft position relative to the Sun

			Returns:
			--------
			Tmax : float
				Spacecraft maximum thrust
			Isp : float
				Spacecraft specific impulse

		"""

		SAA = 0
		Psupply = 13.75
		eff = 0.92

		Pbmp = (-40.558 * r**3 + 173.49 * r**2 -
				259.19 * r + 141.86) * mt.cos(SAA)
		P = -146.26 * r**3 + 658.52 * r**2 - 1059.2 * r + 648.24  # 6 panels
		# P = -195.02 * r**3 + 878.03 * r**2 - 1412.3 * r + 864.32 # 8 panels
		if Pbmp < Psupply:
			P -= (Psupply - Pbmp)
		Pin = eff * P
		if Pin > 120:
			Pin = 120  # thermal max 120W

		Tmax = (26.27127 * Pin - 708.973) / 1000000
		if Tmax < 0:
			Tmax = 0

		Isp = -0.0011 * Pin**3 + 0.175971 * Pin**2 + 4.193797 * Pin + 2037.213

		return Tmax, Isp

	def propagate(self, x):
		""" """

		# Decoding of the decision vector
		t0 = x[0]
		tof = x[1]
		mf = x[2]

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
		throttles = [x[3 + 3 * i: 6  + 3 * i] for i in range(n_seg)]

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

		# Computation of the initial and final epochs and ephemerides
		ti = pk.epoch(t0)
		ri, vi = self.earth.eph(ti)

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
			if self.sep:
				r = math.sqrt(rfwd[i][0]**2 + rfwd[i][1]
							  ** 2 + rfwd[i][2]**2) / pk.AU
				Tmax, isp = self._sep_model(r)
				veff = isp * pk.G0
			ufwd[i] = [Tmax * thr for thr in t]

			rfwd[i + 1], vfwd[i + 1], mfwd[i + 1] = pk.propagate_taylor(
				rfwd[i], vfwd[i], mfwd[i], ufwd[i], fwd_dt[i], MU_SUN, \
				veff, -10, -10)

		# Backward propagation
		bwd_grid = t0 + tof * self.bwd_grid
		bwd_dt = tof * self.bwd_dt

		# Final conditions
		rbwd[-1] = rf
		vbwd[-1] = vf
		mbwd[-1] = mf

		# Propagate
		for i, t in enumerate(throttles[-1:-bwd_seg - 1:-1]):
			if self.sep:
				r = math.sqrt(rbwd[-i - 1][0]**2 + rbwd[-i - 1]
							  [1]**2 + rbwd[-i - 1][2]**2) / pk.AU
				Tmax, isp = self._sep_model(r)
				veff = isp * pk.G0
			ubwd[-i - 1] = [Tmax * thr for thr in t]

			rbwd[-i - 2], vbwd[-i - 2], mbwd[-i - 2] = pk.propagate_taylor(
				rbwd[-i - 1], vbwd[-i - 1], mbwd[-i - 1], ubwd[-i - 1], -bwd_dt[-i - 1], MU_SUN, veff, -10, -10)

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

		throttles = [x[3 + 3 * i: 6 + 3 * i] for i in range(n_seg)]
		alphas = [min(1., np.linalg.norm(t)) for t in throttles]

		times = np.concatenate((fwd_grid, bwd_grid))

		rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, ufwd, ubwd, fwd_dt, bwd_dt, dfwd, dbwd = self.propagate(
			x)

		# Plotting the Sun, the Earth and the target
		axes.scatter([0], [0], [0], color='y')
		pk.orbit_plots.plot_planet(self.earth, pk.epoch(
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
			if self.sep:
				r = math.sqrt(rfwd[i][0]**2 + rfwd[i][1]
							  ** 2 + rfwd[i][2]**2) / pk.AU
				_, isp = self.sep_model(r)
				veff = isp * pk.G0

			pk.orbit_plots.plot_taylor(rfwd[i], vfwd[i], mfwd[i], ufwd[i], fwd_dt[
										   i], MU_SUN, veff, N=10, units=units, color=(alphas[i], 0, 1 - alphas[i]), axes=axes)

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
			if self.sep:
				r = math.sqrt(rbwd[-i - 1][0]**2 + rbwd[-i - 1]
							  [1]**2 + rbwd[-i - 1][2]**2) / pk.AU
				_, isp = self.sep_model(r)
				veff = isp * pk.G0

			else:
				pk.orbit_plots.plot_taylor(rbwd[-i - 1], vbwd[-i - 1], mbwd[-i - 1], ubwd[-i - 1], -bwd_dt[-i - 1],
										   MU_SUN, veff, N=10, units=units, color=(alphas[-i - 1], 0, 1 - alphas[-i - 1]), axes=axes)
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

		throttles = [np.linalg.norm(x[3 + 3 * i: 6 + 3 * i])
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
		if self.sep:
			Tmax = self.sc.thrust
			thrusts = np.linalg.norm(
				np.array(ufwd + ubwd), axis=1) / Tmax
			# plot maximum thrust achievable at that distance from the Sun
			distsSun = dist_sun[:fwd_seg] + \
				dist_sun[-bwd_seg:] + [dist_sun[-1]]
			Tmaxs = [self.sep_model(d)[0] / Tmax for d in distsSun]
			axes.step(np.concatenate(
				(fwd_grid, bwd_grid[1:])), Tmaxs, where="post", c='lightgray', linestyle=':')
		else:
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
		return "Low-Thrust transfer between Earth and NEOs - Preliminary design"

	def get_extra_info(self):
		retval = "\tTarget NEO: " + self.target.name
		retval += "\n\tSolar Electric Propulsion: " + str(self.sep)
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

	def double_segments(self, x):
		"""
		Returns the decision vector encoding a low trust trajectory having double the number of segments with respect to x
		and a 'similar' throttle history. In case high fidelity is True, and x is a feasible trajectory, the returned decision vector
		also encodes a feasible trajectory that can be further optimized

		Parameters:
		-----------
		x : array
			Best decision vector returned by the previous optimization

		Returns:
		--------
		new_prob : <Earth2Asteroid>
			The new udp having twice the segments
		new_x : list
			The new decision vector to be used as initial guess

		"""

		new_x = np.append(x[:3], np.repeat(x[3:].reshape((-1, 3)), 2, axis=0))

		new_prob = Earth2Asteroid(
			target=self.target,
			n_seg=2 * self.n_seg,
			grid_type=self.grid_type,
			t0=[pk.epoch(self.lb[0]), pk.epoch(self.ub[0])],
			tof=[self.lb[1], self.ub[1]],
			m0=self.sc.mass,
			Tmax=self.sc.thrust,
			Isp=self.sc.isp,
			sep=self.sep
		)

		return new_prob, new_x

	def report(self, x):
		"""
		Prints human readable information on the trajectory represented by the decision vector x

		Parameters:
		-----------
		x : array
			Decision vector
	   
		"""

		n_seg = self.n_seg
		mi = self.sc.mass
		t0 = x[0]
		tof = x[1]
		mf = x[2]
		thrusts = [np.linalg.norm(x[3 + 3 * i: 6 + 3 * i])
				   for i in range(n_seg)]

		tf = t0 + tof
		mP = mi - mf
		deltaV = self.sc.isp * pk.G0 * np.log(mi / mf)

		dt = np.append(self.fwd_dt, self.bwd_dt) * tof / pk.DAY2SEC
		time_thrusts_on = sum(dt[i] for i in range(
			len(thrusts)) if thrusts[i] > 0.1)

		print("Departure:", pk.epoch(t0), "(", t0, "mjd2000)")
		print("Time of flight:", tof, "days")
		print("Arrival:", pk.epoch(tf), "(", tf, "mjd2000)")
		print("Delta-v:", deltaV, "m/s")
		print("Propellant consumption:", mP, "kg")
		print("Thrust-on time:", time_thrusts_on, "days")

if __name__ == '__main__':

	from pykep.examples import add_gradient, algo_factory

	# Loading of the 2020-CD3 Kernel
	# pk.util.load_spice_kernel("2020_CD3.bsp")
	# ast = pk.planet.spice('54000953', 'SUN', 'ECLIPJ2000', 'NONE', MU_SUN)
	ast = load_asteroid('2020 CD3')

	# 2 - Launch window
	lw_low = pk.epoch_from_string('2021-01-01 00:00:00')
	lw_upp = pk.epoch_from_string('2021-01-01 00:00:01')

	# 3 - Time of flight
	tof_low = 50
	tof_upp = 5 * 365

	# 4 - Spacecraft
	m0 = 600
	Tmax = 0.23
	Isp = 2700

	# 5 - Optimization algorithm
	algorithm = algo_factory('slsqp')
	algorithm.extract(pg.nlopt).xtol_rel = 1e-8
	algorithm.extract(pg.nlopt).maxeval = 2000

	# 6 - Problem
	udp = Earth2Asteroid(target=ast, n_seg=30, grid_type='uniform', t0=(lw_low, lw_upp), \
		tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, sep=False)

	problem = pg.problem(udp)
	problem.c_tol = [1e-5] * problem.get_nc()

	# 7 - Population
	population = pg.population(problem, size=1, seed=123)

	# 8 - Optimization
	population = algorithm.evolve(population)

	# 9 - Inspect the solution
	print("Feasibility :", problem.feasibility_x(population.champion_x))
	udp.report(population.champion_x)

	# 10 - plot trajectory
	udp.plot_traj(population.champion_x, plot_segments=True)
	plt.title("The trajectory in the heliocentric frame")

	udp.plot_dists_thrust(population.champion_x)

	plt.show()
