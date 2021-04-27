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

from scripts.missions.NEA_Earth.NEA_Earth_Initial_Guess import initial_guess

from scripts.udp.NEA_Earth.NEA_Earth_UDP import NEA2Earth

from scripts.utils.pickle_results import save
from scripts.utils.post_process import post_process
from scripts.utils import load_sqp, load_kernels, load_bodies

""" 

This script runs the optimization of a transfer between a NEA and the Earth using low-thrust propulsion 
using ISAE-SUPAERO super-computers Rainman or Pando. 

Three arguments must be provided to the script when it's runned : 
-----------------------------------------------------------------

	1) The SQP algorithm used (eg. ipopt or slsqp)
	2) The first year of the launch window (eg. 2040)

"""

# First launch window year
year = str(sys.argv[1])

# Loading the main kernels
load_kernels.load()

# Creation of the communicator 
comm = MPI.COMM_WORLD
rank = comm.rank

# - * - * - * - * - * - * - * - * - * - * - * - * 
print("Rank <{}> : Run".format(rank), flush=True)
# - * - * - * - * - * - * - * - * - * - * - * - * 

# Year of interest
year = int(year)

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
tof_low = cst.YEAR2DAY * 0.70
tof_upp = cst.YEAR2DAY * 5.00

# 4 - Spacecraft
# --------------
m0 = 2000 + ast_mass
Tmax = 0.5
Isp = 3000

# 5 - Earth arrival 
# -----------------
vinf_max = 2.5e3

# 6 - Optimization algorithm
# --------------------------
algorithm = load_sqp.load('ipopt')

# 7 - Problem
# -----------
n_seg = 30

udp = NEA2Earth(nea=ast, n_seg=n_seg, t0=(lw_low, lw_upp), tof=(tof_low, tof_upp), m0=m0, \
	Tmax=Tmax, Isp=Isp, nea_mass=ast_mass, vinf_max=vinf_max, earth_grv=True)
problem = pg.problem(udp)

# 7 - Population
# --------------
population = pg.population(problem, size=1)

# 8 - Initial guess generation
# ----------------------------

xi = initial_guess(year_=year, n_seg=n_seg)
population.set_x(0, xi)

# 9 - Optimization
# ----------------
population = algorithm.evolve(population)
x = population.get_x()[0]

# 10 - Pickle the results
# -----------------------
host = 'rainman' if 'semblanet' in getpass.getuser() else 'pando'
save(host=host, mission='NEA_Earth', udp=udp, population=population)
