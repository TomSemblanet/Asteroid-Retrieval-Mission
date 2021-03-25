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

from scripts.utils.loaders import load_sqp, load_kernels, load_bodies

from scripts.UDP.Earth2NEA_UDP import Earth2NEA

from scripts.utils.post_process import post_process

from data import constants as cst


# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')

# 2 - Launch window
lw_low = pk.epoch_from_string('2021-01-01 00:00:00')
lw_upp = pk.epoch_from_string('2021-12-31 23:59:59')

# 3 - Time of flight
tof_low = cst.YEAR2DAY * 0.1
tof_upp = cst.YEAR2DAY * 5.00

# 4 - Spacecraft
m0 = 600
Tmax = 0.5
Isp = 3000

# 5 - Velocity at infinity
vinf_max = 2e3

# 5 - Optimization algorithm
algorithm = load_sqp.load('slsqp')
algorithm.extract(pg.nlopt).maxeval = 200 # /!\ /!\ NOT TO HIGH (200 is OK) /!\ /!\ 

# 7 - Problem
udp = Earth2NEA(nea=ast, n_seg=100, t0=(lw_low, lw_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max)

problem = pg.problem(udp)
problem.c_tol = [1e-8] * problem.get_nc()

# 8 - Population
population = pg.population(problem, size=1, seed=200)

# 9 - Optimization
population = algorithm.evolve(population)

udp.check_con_violation(population.champion_x)

res = {'udp': udp, 'population': population}

with open('res', 'wb') as f:
	pkl.dump(res, f)

# 10 - Post process
post_process(udp, population.champion_x)

