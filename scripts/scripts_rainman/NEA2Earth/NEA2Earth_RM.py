#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 2021 09:02:20

@author: SEMBLANET Tom

"""

import os
import sys
import getpass

import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl

from scripts.utils.loaders import load_sqp, load_kernels, load_bodies
from scripts.UDP.NEA2Earth_UDP import NEA2Earth
from scripts.utils.post_process import post_process
from data import constants as cst


# Range of launch years from user input arguments
year_l = int(sys.argv[1])
year_u = int(sys.argv[2])

# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')
ast_mass = 4900 # 2020-CD3 mass [kg]

# Launch years
launch_years = np.arange(year_l, year_u)


# Optimization of the trajectory on various launch window
for year in launch_years:

	# Launch window
	lw_low = pk.epoch_from_string(str(year)+'-01-01 00:00:00')
	lw_upp = pk.epoch_from_string(str(year)+'-12-31 23:59:59')

	# Time of flight
	tof_low = cst.YEAR2DAY * 0.1
	tof_upp = cst.YEAR2DAY * 5.00

	# Spacecraft
	m0 = 2000 + ast_mass
	Tmax = 0.5
	Isp = 3000

	# Optimization algorithm
	algorithm = load_sqp.load('slsqp')
	algorithm.extract(pg.nlopt).maxeval = 200

	# Problem
	udp = NEA2Earth(nea=ast, n_seg=30, t0=(lw_low, lw_upp), \
		tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, nea_mass=ast_mass)

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
		

	# Pickle of the results
	res = {'udp': udp, 'population': population}


	if 'gary' in getpass.getuser():
		storage_path = '/scratch/dcas/yv.gary/SEMBLANET/NEA_Earth_results/500/NEA_Earth_500_' + str(year)
	else:
		storage_path = '/scratch/students/t.semblanet/NEA_Earth_results/500/NEA_Earth_500_' + str(year)

	with open(storage_path, 'wb') as f:
		pkl.dump(res, f)
