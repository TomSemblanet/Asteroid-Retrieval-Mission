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


from scripts.udp.NEA_Earth_UDP import NEA2Earth

from scripts.missions.NEA_Earth.NEA_Earth_Initial_Guess import initial_guess

from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.utils.post_process import post_process
from scripts.utils.pickle_results import save

from data import constants as cst

# Year of interest
year = int(sys.argv[1])

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
tof_upp = cst.YEAR2DAY * 3.00

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

# 5 - Optimization algorithm
# --------------------------
algorithm = load_sqp.load('ipopt')

# 6 - Problem
# -----------
n_seg = 30

udp = NEA2Earth(nea=ast, n_seg=n_seg, t0=(lw_low, lw_upp), tof=(tof_low, tof_upp), m0=m0, \
	Tmax=Tmax, Isp=Isp, nea_mass=ast_mass, phi_min=phi_min, phi_max=phi_max, theta_min=theta_min, \
	theta_max=theta_max, earth_grv=True)
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
# * - * - * - * - * - * - * - * - * - 
print("Main ptimization")
# * - * - * - * - * - * - * - * - * - 
population = algorithm.evolve(population)
x = population.get_x()[0]

# 10 - Post process 
# -----------------
post_process(udp, x)

# 12 - Pickle the results
# -----------------------
save(host='laptop', mission='NEA_Earth', udp=udp, population=population)


