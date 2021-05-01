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
from datetime import date

from mpi4py import MPI

from data import constants as cst
from data.nea_mass_computation import get_mass

from scripts.udp.NEA_Earth_UDP import NEA2Earth

from scripts.utils.post_process import post_process
from scripts.utils import load_sqp, load_kernels, load_bodies

""" 

This scripts runs multiple optimization of a transfer between a given NEA (eg. CD3-2020) and the Earth
using low-fidelity model. Its aim is to give a rough idea of the average deltaV for a NEA -> Earth transfer
for a departure at a given year (2025 - 2050).

"""

# Loading the main kernels
load_kernels.load()

# Creation of the communicator 
comm = MPI.COMM_WORLD
rank = comm.rank

# - * - * - * - * - * - * - * - * - * - * - * - * 
print("Rank <{}> : Run for year {}".format(rank, 2025 + rank), flush=True)
# - * - * - * - * - * - * - * - * - * - * - * - * 

# Year of interest
year = 2025 + int(rank)

# Loading the main kernels
load_kernels.load()

# 1 - Asteroid data
# -----------------
# ast = load_bodies.asteroid('2020 CD3')
# ast_mass = 4900 
ast = load_bodies.asteroid('2018 WV1')
ast_mass = get_mass(H=30.145)

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
algorithm = load_sqp.load('slsqp')

# 7 - Problem
# -----------
udp = NEA2Earth(nea=ast, n_seg=20, t0=(lw_low, lw_upp), tof=(tof_low, tof_upp), m0=m0, \
	Tmax=Tmax, Isp=Isp, nea_mass=ast_mass, phi_min=phi_min, phi_max=phi_max, theta_min=theta_min, \
	theta_max=theta_max, earth_grv=True)
problem = pg.problem(udp)

# 7 - Population
# --------------
population = pg.population(problem, size=1)

# 8 - DeltaV computation
# ----------------------

# Sum of deltaVs
sum_ = 0

# Number of iterations
N = 50
count = 0

while count < N:
	# Generation of a random decision vector
	x = population.random_decision_vector()

	# Set the decision vector
	population.set_x(0, x)

	# Optimization
	population = algorithm.evolve(population)
	x = population.get_x()[0]

	# Adding the deltaV to the sum [m/s]
	sum_ += udp.get_deltaV(x)

	count += 1

# Computation of the average deltaV
average_dV = sum_ / N

# Location and name of the results file
path = str(sys.argv[1])

# Write the file
file = open(path, 'a')
file.write("Year {} : {}\n".format(year, average_dV))

# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
print("Year {}, results stored in {}".format(year, path), flush=True)
# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *

