#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 2021 09:02:20

@author: SEMBLANET Tom

"""

import pykep as pk 
import pygmo as pg 
import pickle as pkl

from scripts.loaders import load_sqp, load_kernels, load_bodies

from scripts.UDP.NEA2Earth_UDP import NEA2Earth

from scripts.post_process import post_process

from data import constants as cst


# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')

# 2 - Launch window
lw_low = pk.epoch_from_string('2039-01-01 00:00:00')
lw_upp = pk.epoch_from_string('2045-12-31 23:59:59')

# 3 - Time of flight
tof_low = cst.YEAR2DAY * 0.01
tof_upp = cst.YEAR2DAY * 5.00

# 4 - Spacecraft
m0 = 2000
Tmax = 1
Isp = 2500

# 5 - Asteroid sample mass
mass = 0

# 5 - Optimization algorithm
algorithm = load_sqp.load('slsqp')
algorithm.extract(pg.nlopt).maxeval = 2500

# 6 - Monotonic Basin Hopping method
mbh = pg.algorithm(pg.mbh(algo=algorithm))
mbh.set_verbosity(3)

# 6 - Problem
udp = NEA2Earth(nea=ast, n_seg=20, t0=(lw_low, lw_upp), \
	tof=(tof_low, tof_upp), m0=(m0+mass), Tmax=Tmax, Isp=Isp)

problem = pg.problem(udp)
problem.c_tol = [1e-8] * problem.get_nc()

# 7 - Population
population = pg.population(problem, size=1, seed=123)

# 8 - Optimization
population = algorithm.evolve(population)
# population = mbh.evolve(population)

# 10 - Post process
post_process(udp, population.champion_x)