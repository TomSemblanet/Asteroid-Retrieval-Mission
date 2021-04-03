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

from scripts.utils import load_sqp, load_kernels, load_bodies

from scripts.udp.Earth2NEA_UDP import Earth2NEA

from scripts.utils.post_process import post_process

from data import constants as cst


# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')

# Minimum stay time [days]
stay_time = 90

# 2 - Arrival date
arr_low = pk.epoch(15801.132 - 2 * cst.YEAR2DAY, 'mjd2000')
arr_upp = pk.epoch(15801.132 - stay_time, 'mjd2000')

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
algorithm = load_sqp.load('ipopt')

# 7 - Problem
udp = Earth2NEA(nea=ast, n_seg=30, tf=(arr_low, arr_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max, earth_grv=True)
problem = pg.problem(udp)

# 7 - Population
# --------------
population = pg.population(problem, size=1)

# 8 - Starting point
# ------------------
# Number of iterations
N = 100
count = 0

found_sol = False

# Best decision-vector
x_best = population.get_x()[0]

while count < N:
	# Generation of a random decision vector
	x = population.random_decision_vector()

	# Generate random decision vector until one provides a good starting point
	while (-udp.fitness(x)[0] < 0.98) :
		x = population.random_decision_vector()

	# Set the decision vector
	population.set_x(0, x)

	# Optimization
	population = algorithm.evolve(population)

	# Mismatch error on position [km] and velocity [km/s]
	error_pos = np.linalg.norm(udp.fitness(population.champion_x)[1:4]) * pk.AU / 1000
	error_vel = np.linalg.norm(udp.fitness(population.champion_x)[4:7]) * pk.EARTH_VELOCITY / 1000

	# Update the best decision vector found
	if (-udp.fitness(x)[0] > -udp.fitness(x_best)[0] and error_pos < 10e3 and error_vel < 0.01):
		x_best = x 
		found_sol = True

	count += 1

# 10 - Check constraints violation
udp.check_con_violation(population.champion_x)

# 11 - Post process the results
post_process(udp, population.champion_x)

# 12 - Pickle the results
res = {'udp': udp, 'population': population}
with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/03_04_2021_results/' + str(year), 'wb') as f:
	pkl.dump(res, f)
