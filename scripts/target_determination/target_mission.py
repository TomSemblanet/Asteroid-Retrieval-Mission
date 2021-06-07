#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 07 2021 18:38:20

@author: SEMBLANET Tom

"""

import os
import sys
import getpass

import matplotlib.pyplot as plt

import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl
from datetime import date

from mpi4py import MPI

from data import constants as cst
from data.spk_table import NAME2SPK
from scripts.utils.post_process import post_process
from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.utils.pickle_results import save
from scripts.target_determination.target_udp import Earth2NEA

# Number of the script
script_number = int(sys.argv[1])

# Creation of the communicator 
comm = MPI.COMM_WORLD
rank = comm.rank

# - * - * - * - * - * - * - * - * - * - * - * - * 
print("Rank <{}> : Run".format(rank), flush=True)
# - * - * - * - * - * - * - * - * - * - * - * - * 

# 1 - Load of the SPICE kernels
load_kernels.load()

# 2 - Load of the targets list
# print(list(NAME2SPK.keys())[(script_number-1)*24 + rank])
asteroid = load_bodies.asteroid(list(NAME2SPK.keys())[(script_number-1)*24 + rank])

# 3 - Arrival date 
ad_low = pk.epoch_from_string('2025-01-01 00:00:00')
ad_upp = pk.epoch_from_string('2045-12-31 23:59:59')

# 4 - Time of flight
tof_low = cst.YEAR2DAY * 0.50
tof_upp = cst.YEAR2DAY * 5.00

# 5 - Spacecraft
m0 = 10000
Tmax = 2
Isp = 3000

# 5 - Optimization algorithm
algorithm = load_sqp.load('ipopt', max_iter=3)

# 6 - Problem
udp = Earth2NEA(
        nea=asteroid,
        n_seg=30,
        tf=[ad_low, ad_upp],
        tof=[tof_low, tof_upp],
        m0=m0,
        Tmax=Tmax,
        Isp=Isp,
        earth_grv=False)

problem = pg.problem(udp)

# 7 - Population
population = pg.population(problem, size=1, seed=1)

# 8 - Optimization
population = algorithm.evolve(population)
x = population.get_x()[0]

# 10 - Post process 
# -----------------
dV = udp.get_deltaV(x)

with open('/home/dcas/yv.gary/SEMBLANET/Asteroid-Retrieval-Mission/local/target_determination/deltaV.txt', 'a') as file:
    file.write(list(NAME2SPK.keys())[(script_number-1)*24 + rank] + ' ' + str(dV) + '\n')
