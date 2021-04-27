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
from datetime import date

from mpi4py import MPI

from data import constants as cst
from scripts.udp.Earth_NEA.Earth_NEA_UDP import Earth2NEA
from scripts.utils.post_process import post_process
from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.missions.Earth_NEA.Earth_NEA_Initial_Guess import initial_guess


# NEA Departure day
nea_dpt_date = float(sys.argv[1])

# Maximum dV [m/s]
dV_max = float(sys.argv[2])

# Creation of the communicator 
comm = MPI.COMM_WORLD
rank = comm.rank

# - * - * - * - * - * - * - * - * - * - * - * - * 
print("Rank <{}> : Run".format(rank), flush=True)
# - * - * - * - * - * - * - * - * - * - * - * - * 

# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')

# Maximum and minimum stay time [days]
max_stay_time = 365
min_stay_time = 90

# 2 - Arrival date
arr_low = pk.epoch(nea_dpt_date - max_stay_time, 'mjd2000')
arr_upp = pk.epoch(nea_dpt_date - min_stay_time, 'mjd2000')

# 3 - Time of flight
tof_low = cst.YEAR2DAY * 0.50
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
n_seg = 30
udp = Earth2NEA(nea=ast, n_seg=n_seg, tf=(arr_low, arr_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max, earth_grv=True)
problem = pg.problem(udp)

# 7 - Population
# --------------
population = pg.population(problem, size=1)

# 8 - Optimization
# ----------------
xi = initial_guess(nea_dpt_date_=nea_dpt_date, n_seg=n_seg)

# Set the initial decision vector
population.set_x(0, xi)

# 9 - Optimization
# ----------------
# * - * - * - * - * - * - * - * - * - 
print("Main optimization", flush=True)
# * - * - * - * - * - * - * - * - * - 
population = algorithm.evolve(population)
x = population.get_x()[0]

# 10 - Post process 
# -----------------
post_process(udp, x)

# 11 - Pickle the results
# -----------------------
dV = udp.get_deltaV(x)

if dV <= dV_max:
	host = 'rainman' if 'semblanet' in getpass.getuser() else 'pando'
	save(host='laptop', mission='Earth_NEA', udp=udp, population=population)
else:
	# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - 
	print("Delta-V : {} m/s > {} m/s".format(dV, dV_max), flush=True)
	# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - 
