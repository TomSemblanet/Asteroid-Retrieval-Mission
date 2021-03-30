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

# Creation of the communicator 
comm = MPI.COMM_WORLD
rank = comm.rank

print("Process rank <{}> launched".format(rank), flush=True)
# sys.stdout.write('{} launched'.format(rank))

# Initial year
year_i = int(sys.argv[1])

# Number of allocated processors
n_proc = int(sys.argv[2])

# Year of NEA departure
year = year_i + comm.rank

# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')
ast_mass = 4900 # 2020-CD3 mass [kg]

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
algorithm.set_verbosity(0)

# Problem
udp = NEA2Earth(nea=ast, n_seg=10, t0=(lw_low, lw_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, nea_mass=ast_mass)

problem = pg.problem(udp)
problem.c_tol = [1e-8] * problem.get_nc()

pos_err = 1e10
dV = 1e10


 
while pos_err > 5000 or dV > 1000:

	seed = np.random.randint(1, 100000)
	print("<{}> : {}".format(rank, seed), flush=True)

	# Population
	population = pg.population(problem, size=1, seed=seed)

	# Optimization
	population = algorithm.evolve(population)

	# Check feasibility
	fitness = udp.fitness(population.champion_x)

	pos_err = np.linalg.norm(fitness[1:4]) * pk.AU / 1000
	dV = udp.sc.isp * cst.G0 * np.log(- 1 / fitness[0])

print("*** Solution found for year {} ***".format(year), flush=True)

# Pickle of the results
res = {'udp': udp, 'population': population}


if 'gary' in getpass.getuser():
	storage_path = '/scratch/dcas/yv.gary/SEMBLANET/NEA_Earth_results/500/NEA_Earth_500_' + str(year)
else:
	storage_path = '/scratch/students/t.semblanet/NEA_Earth_results/500/NEA_Earth_500_' + str(year)

with open(storage_path, 'wb') as f:
	pkl.dump(res, f)

