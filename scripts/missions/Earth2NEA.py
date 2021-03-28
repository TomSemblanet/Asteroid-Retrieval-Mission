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
lw_low = pk.epoch_from_string('2026-01-01 00:00:00')
lw_upp = pk.epoch_from_string('2026-12-31 23:59:59')

# 3 - Time of flight
tof_low = cst.YEAR2DAY * 0.1
tof_upp = cst.YEAR2DAY * 3.00

# 4 - Spacecraft
m0 = 2000
Tmax = 0.5
Isp = 3000

# 5 - Velocity at infinity
vinf_max = 2e3

# 5 - Optimization algorithm
algorithm = load_sqp.load('slsqp')
algorithm.extract(pg.nlopt).maxeval = 200

# 7 - Problem
udp_no_earth_g = Earth2NEA(nea=ast, n_seg=50, t0=(lw_low, lw_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max, earth_grv=False)

problem = pg.problem(udp_no_earth_g)
problem.c_tol = [1e-8] * problem.get_nc()

# 8 - Population
population = pg.population(problem, size=1, seed=200)

# 9 - Optimization
population = algorithm.evolve(population)

# 10 - Check constraints violation
udp_no_earth_g.check_con_violation(population.champion_x)

# 11 - Post process the results
post_process(udp_no_earth_g, population.champion_x)

# BONUS
udp_earth_g = Earth2NEA(nea=ast, n_seg=50, t0=(lw_low, lw_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max, earth_grv=True)

problem = pg.problem(udp_earth_g)
problem.c_tol = [1e-8] * problem.get_nc()

population2 = pg.population(problem, size=0)
population2.push_back(population.champion_x)

population2 = algorithm.evolve(population2)

# 10 - Check constraints violation
udp_earth_g.check_con_violation(population2.champion_x)

# 11 - Post process the results
post_process(udp_earth_g, population2.champion_x)


