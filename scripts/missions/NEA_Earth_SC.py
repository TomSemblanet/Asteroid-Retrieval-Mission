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

from mpi4py import MPI

from data import constants as cst
from scripts.udp.NEA2Earth_UDP import NEA2Earth
from scripts.utils.post_process import post_process
from scripts.utils import load_sqp, load_kernels, load_bodies

""" 

This scripts runs the optimization of a transfer between a NEA and the Earth using low-thrust propulsion 
using ISAE-SUPAERO super-computers Rainman or Pando. 

"""

# SQP algorithm
sqp = str(sys.argv[1])

# Loading the main kernels
load_kernels.load()

# Creation of the communicator 
comm = MPI.COMM_WORLD
rank = comm.rank

# - * - * - * - * - * - * - * - * - * - * - * - * 
print("Rank <{}> : Run".format(rank), flush=True)
# - * - * - * - * - * - * - * - * - * - * - * - * 

# Year of interest
year = 2026 + int(rank)

# Loading the main kernels
load_kernels.load()

# 1 - Asteroid data
# -----------------
ast = load_bodies.asteroid('2020 CD3')
ast_mass = 4900 

# 2 - Launch window
# -----------------
lw_low = pk.epoch_from_string(str(year) + '-01-01 00:00:00')
lw_upp = pk.epoch_from_string(str(year) + '-12-31 23:59:59')

# 3 - Time of flight
# ------------------
tof_low = cst.YEAR2DAY * 0.1
tof_upp = cst.YEAR2DAY * 5.00

# 4 - Spacecraft
# --------------
m0 = 2000 + ast_mass
Tmax = 0.5
Isp = 3000

# 5 - Earth arrival 
# -----------------
phi_min = 175.0 * cst.DEG2RAD
phi_max = 185.0 * cst.DEG2RAD

theta_min = 89.0 * cst.DEG2RAD
theta_max = 91.0 * cst.DEG2RAD

# 6 - Optimization algorithm
# --------------------------
algorithm = load_sqp.load(sqp)

# 7 - Problem
# -----------
udp = NEA2Earth(nea=ast, n_seg=20, t0=(lw_low, lw_upp), tof=(tof_low, tof_upp), m0=m0, \
	Tmax=Tmax, Isp=Isp, nea_mass=ast_mass, phi_min=phi_min, phi_max=phi_max, theta_min=theta_min, \
	theta_max=theta_max, earth_grv=True)
problem = pg.problem(udp)

# 7 - Population
# --------------
population = pg.population(problem, size=1)

# 8 - Starting point
# ------------------
# Number of iterations
N = 50
count = 0

# Best decision-vector
x_best = population.get_x()[0]

while count < N:
	# Generation of a random decision vector
	x = population.random_decision_vector()

	# Generate random decision vector until one provides a good starting point
	while -udp.fitness(x)[0] < 0.90:
		x = population.random_decision_vector()

	# Set the decision vector
	population.set_x(0, x)

	# Optimization
	population = algorithm.evolve(population)
	x = population.get_x()[0]

	# Mismatch error on position [km] and velocity [km/s]
	error_pos = np.linalg.norm(udp.fitness(x)[1:4]) * pk.AU / 1000
	error_vel = np.linalg.norm(udp.fitness(x)[4:7]) * pk.EARTH_VELOCITY / 1000

	# Update the best decision vector found
	if (-udp.fitness(x)[0] > -udp.fitness(x_best)[0] and error_pos < 100e3 and error_vel < 0.05):
		x_best = x
		found_sol = True

	count += 1

# Keep the best decision vector found
population.set_x(0, x_best)

# 11 - Pickle the results
if found_sol == True:
	res = {'udp': udp, 'population': population}
	with open('/scratch/students/t.semblanet/NEA_Earth_results/' + str(sqp) + '/' + str(year), 'wb') as f:
		pkl.dump(res, f)

	# - * - * - * - * - * - * - * - * - * - * - * - * 
	print("Rank <{}> : Finished successfully!".format(rank), flush=True)
	# - * - * - * - * - * - * - * - * - * - * - * - * 

else:
	# - * - * - * - * - * - * - * - * - * - * - * - * 
	print("Rank <{}> : Finished with failure.".format(rank), flush=True)
	# - * - * - * - * - * - * - * - * - * - * - * - * 



