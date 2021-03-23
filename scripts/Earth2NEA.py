#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 2021 09:02:20

@author: SEMBLANET Tom

"""

import os
import pykep as pk 
import pygmo as pg 
import pickle as pkl
from datetime import datetime as dt

from scripts.loaders import load_sqp, load_kernels, load_bodies

from scripts.UDP.Earth2NEA_UDP import Earth2NEA

from scripts.post_process import post_process

from data import constants as cst


# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')

# 2 - Launch window
lw_low = pk.epoch_from_string('2020-01-01 00:00:00')
lw_upp = pk.epoch_from_string('2020-12-31 23:59:59')

# 3 - Time of flight
tof_low = cst.YEAR2DAY * 0.01
tof_upp = cst.YEAR2DAY * 4.00

# 4 - Spacecraft
m0 = 600
Tmax = 1
Isp = 2700

# 5 - Velocity at infinity
vinf_max = 2.5e3

# 5 - Optimization algorithm
algorithm = load_sqp.load('slsqp')
algorithm.extract(pg.nlopt).maxeval = 2

# 6 - Monotonic Basin Hopping method
mbh = pg.algorithm(pg.mbh(algo=algorithm))
mbh.set_verbosity(5)

# 7 - Problem
udp = Earth2NEA(nea=ast, n_seg=30, t0=(lw_low, lw_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max)

problem = pg.problem(udp)
problem.c_tol = [1e-8] * problem.get_nc()

# 8 - Population
population = pg.population(problem, size=1, seed=123)

# 9 - Optimization
algorithm.evolve(population)
# population = mbh.evolve(population)

# 10 - Post process
post_process(udp, population.champion_x)