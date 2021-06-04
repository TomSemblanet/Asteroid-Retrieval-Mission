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

import matplotlib.pyplot as plt

from data import constants as cst
from data.nea_mass_computation import get_mass

from scripts.udp.NEA_Earth.NEA_Earth_UDP import NEA2Earth

from scripts.utils.pickle_results import save
from scripts.utils.post_process import post_process
from scripts.utils import load_sqp, load_kernels, load_bodies

# Path of the text file containing the decision vector of interest
file_path = sys.argv[1]

# Read the content of the file
file = open(file_path, 'r')
lines = file.readlines()

# Create the decision vector
x = np.array([])
for line in lines:
	x = np.append(x, float(line[:-2]))

# Loading the main kernels
load_kernels.load()

# 1 - Asteroid data
# -----------------
ast = load_bodies.asteroid('2020 CD3')
ast_mass = 4900 

# 2 - Launch window
# -----------------
lw_low = pk.epoch_from_string('2043-01-01 00:00:00')
lw_upp = pk.epoch_from_string('2044-12-31 23:59:59')

# 3 - Time of flight
# ------------------
tof_low = cst.YEAR2DAY * 0.70
tof_upp = cst.YEAR2DAY * 4.00

# 4 - Spacecraft
# --------------
m0 = 10000 + ast_mass
Tmax = 2
Isp = 3000

# 5 - Earth arrival 
# -----------------
vinf_max = 3.5e3

# 6 - Problem
# -----------
udp = NEA2Earth(nea=ast, n_seg=120, t0=(lw_low, lw_upp), tof=(tof_low, tof_upp), m0=m0, \
	Tmax=Tmax, Isp=Isp, nea_mass=ast_mass, vinf_max=vinf_max, earth_grv=True)
problem = pg.problem(udp)

# 7 - Population
# --------------
population = pg.population(problem, size=1)

# Update the best result
population.set_x(0, x)

# Inspect the solution
post_process(udp, population.get_x()[0])


# Pickle the results
# ------------------
save(host='laptop', mission='NEA_Earth', udp=udp, population=population)




