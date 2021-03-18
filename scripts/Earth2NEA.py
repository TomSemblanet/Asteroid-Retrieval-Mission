#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 2021 09:02:20

@author: SEMBLANET Tom

"""

import pykep as pk 
import pygmo as pg 

from scripts.loaders.load_kernels import load_kernels
from scripts.loaders.load_bodies import load_asteroid

from data import constants as cst


# Loading the main kernels
load_kernels()

# Loading of the target asteroid
ast = load_asteroid('2020 CD3')

# 2 - Launch window
lw_low = pk.epoch_from_string('2021-01-01 00:00:00')
lw_upp = pk.epoch_from_string('2030-01-01 00:00:01')

# 3 - Time of flight
tof_low = cst.YEAR2DAY * 0.1
tof_upp = cst.YEAR2DAY * 5.0

# 4 - Spacecraft
m0 = 600
Tmax = 0.23
Isp = 2700

# 5 - Velocity at infinity
vinf_min = 0
vinf_max = 1.5e3

# 5 - Optimization algorithm
algorithm = algo_factory('slsqp')
algorithm.extract(pg.nlopt).xtol_rel = 1e-8
algorithm.extract(pg.nlopt).maxeval = 3000

# 6 - Problem
udp = Earth2Asteroid(target=ast, n_seg=30, grid_type='uniform', t0=(lw_low, lw_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf=[vinf_min, vinf_max])

problem = pg.problem(udp)
problem.c_tol = [1e-5] * problem.get_nc()

# 7 - Population
population = pg.population(problem, size=1, seed=123)

# 8 - Optimization
population = algorithm.evolve(population)

# 9 - Inspect the solution
print("Feasibility :", problem.feasibility_x(population.champion_x))
udp.report(population.champion_x)

# # 10 - plot trajectory
# udp.plot_traj(population.champion_x, plot_segments=True)
# plt.title("The trajectory in the heliocentric frame")

# udp.plot_dists_thrust(population.champion_x)

# plt.show()