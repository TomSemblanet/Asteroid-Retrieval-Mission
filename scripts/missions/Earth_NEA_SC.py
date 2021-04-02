#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 2021 09:02:20

@author: SEMBLANET Tom

"""

import os
import sys
import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl

from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.udp.Earth2NEA_UDP import Earth2NEA
from scripts.utils.post_process import post_process
from data import constants as cst


# Range of launch years from user input arguments
year_l = int(sys.argv[1])
year_u = int(sys.argv[2])

# Spacecraft thrust [mN]
thrust = int(sys.argv[3])

# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')

# Launch years
launch_years = np.arange(year_l, year_u)


# Optimization of the trajectory on various launch window
for year in launch_years:

	# Launch window
	lw_low = pk.epoch_from_string(str(year)+'-01-01 00:00:00')
	lw_upp = pk.epoch_from_string(str(year)+'-12-31 23:59:59')

	# Time of flight
	tof_low = cst.YEAR2DAY * 0.1
	tof_upp = cst.YEAR2DAY * 3.00

	# Spacecraft
	m0 = 2000
	Tmax = thrust / 1000
	Isp = 3000

	# Velocity at infinity (LGA)
	vinf_max = 2e3

	# Optimization algorithm
	algorithm = load_sqp.load('slsqp')
	algorithm.extract(pg.nlopt).maxeval = 200

	# Problem
	udp = Earth2NEA(nea=ast, n_seg=50, t0=(lw_low, lw_upp), \
		tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max)

	problem = pg.problem(udp)
	problem.c_tol = [1e-8] * problem.get_nc()

	# Population
	seed=200
	
	seed_ok = False
	while seed_ok == False:
		try:
			population = pg.population(problem, size=1, seed=seed)
			population = algorithm.evolve(population)
			seed_ok = True
		except:
			seed += 100

	# Pickle of the results
	res = {'udp': udp, 'population': population}

	with open('/scratch/students/t.semblanet/Earth_NEA_results/' + str(thrust) + '/Earth_NEA_'+ str(thrust) + '_'+str(year), 'wb') as f:
		pkl.dump(res, f)

