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

from data import constants as cst


# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')

# 2 - Launch window
lw_low = pk.epoch_from_string('2021-01-01 00:00:00')
lw_upp = pk.epoch_from_string('2059-12-31 23:59:59')

# 3 - Time of flight
tof_low = cst.YEAR2DAY * 0.01
tof_upp = cst.YEAR2DAY * 3.00

# 4 - Spacecraft
m0 = 600
Tmax = 0.23
Isp = 2700

# 5 - Asteroid sample mass
mass = 100e6

# 5 - Maximal velocity magnitude at Moon arrival
v_arr_max = 2.5e3

# 5 - Optimization algorithm
algorithm = load_sqp.load('slsqp')
algorithm.extract(pg.nlopt).maxeval = 2500

# 6 - Problem
udp = Earth2NEA(target=ast, n_seg=30, t0=(lw_low, lw_upp), \
	tof=(tof_low, tof_upp), m0=(m0+mass), Tmax=Tmax, Isp=Isp, v_arr_max=v_arr_max)

problem = pg.problem(udp)
problem.c_tol = [1e-8] * problem.get_nc()

# 7 - Population
population = pg.population(problem, size=1, seed=123)

# 8 - Optimization
population = algorithm.evolve(population)

# If we are on RAINMAIN, we pickle the results to inspect them further
if 'node' in os.uname()[1]:
	rs = {'udp': udp, 'x': population.champion_x}

	date = dt.now().strftime("%d_%m_%Y_%H_%M_%S")
	with open('Earth_NEA_' + str(date), 'wb') as file:
		pkl.dump(obj=rs, file=file, protocol=4)

else:
	# 9 - Inspect the solution
	udp.report(population.champion_x)

	# 10 - Plot the trajectory
	udp.plot_traj(population.champion_x)

	# 11 - Plot the thrust profil
	udp.plot_thrust(population.champion_x)