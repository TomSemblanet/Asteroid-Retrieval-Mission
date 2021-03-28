#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 2021 09:02:20

@author: SEMBLANET Tom

"""

import os
import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl
from datetime import datetime as dt

from scripts.utils.loaders import load_sqp, load_kernels, load_bodies

from scripts.UDP.NEA2Earth_UDP import NEA2Earth

from scripts.utils.post_process import post_process

from data import constants as cst

# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')
ast_mass = 4900 # 2020-CD3 mass [kg]

# 2 - Launch window
lw_low = pk.epoch_from_string('2021-01-01 00:00:00')
lw_upp = pk.epoch_from_string('2021-12-31 23:59:59')

# 3 - Time of flight
tof_low = cst.YEAR2DAY * 0.1
tof_upp = cst.YEAR2DAY * 5.00

# 4 - Spacecraft
m0 = 2000 + ast_mass
Tmax = 0.5
Isp = 3000

# 5 - Optimization algorithm
algorithm = load_sqp.load('slsqp')
algorithm.extract(pg.nlopt).maxeval = 200

# 7 - Problem
udp = NEA2Earth(nea=ast, n_seg=30, t0=(lw_low, lw_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, nea_mass=ast_mass, earth_grv=True)

problem = pg.problem(udp)
problem.c_tol = [1e-8] * problem.get_nc()

error = 1e10
mass_fuel = 1e10

while error > 5000:

	seed = np.random.randint(1, 100000)
	print("Try with seed : <{}>".format(seed))

	# 8 - Population
	population = pg.population(problem, size=1, seed=seed)

	# 9 - Optimization
	population = algorithm.evolve(population)

	# 10 - Check feasibility
	fitness = udp.fitness(population.champion_x)

	error = np.linalg.norm(fitness[1:4]) * pk.AU / 1000

	print("\nError : {} km".format(error))

print("Found a quasi-feasible solution with seed <{}>\n\n".format(seed))

# 10 - Check constraints violation
udp.check_con_violation(population.champion_x)

# 11 - Post process the results
post_process(udp, population.champion_x)
