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

from scripts.udp.Earth_NEA.Earth_NEA_UDP import Earth2NEA

from scripts.utils.pickle_results import save
from scripts.utils.post_process import post_process
from scripts.utils import load_sqp, load_kernels, load_bodies

# Path of the text file containing the decision vector of interest
file_path = sys.argv[1]

# NEA departure date
nea_dpt_date = float(sys.argv[2])

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

# 2 - Launch window
# -----------------
# Maximum and minimum stay time [days]
max_stay_time = 365
min_stay_time = 90

arr_low = pk.epoch(nea_dpt_date - max_stay_time, 'mjd2000')
arr_upp = pk.epoch(nea_dpt_date - min_stay_time, 'mjd2000')

# 3 - Time of flight
# ------------------
tof_low = cst.YEAR2DAY * 0.50
tof_upp = cst.YEAR2DAY * 3.00

# 4 - Spacecraft
# --------------
m0 = 2000
Tmax = 0.5
Isp = 3000

# 5 - Earth arrival 
# -----------------
vinf_max = 2e3

# 6 - Problem
# -----------
n_seg = 120
udp = Earth2NEA(nea=ast, n_seg=n_seg, tf=(arr_low, arr_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max, earth_grv=True)
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
save(host='laptop', mission='Earth_NEA', udp=udp, population=population)




