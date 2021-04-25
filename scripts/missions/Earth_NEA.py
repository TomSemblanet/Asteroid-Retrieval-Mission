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

from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.utils.post_process import post_process

from scripts.udp.Earth_NEA_UDP import Earth2NEA
from scripts.missions.Earth_NEA_Initial_Guess import initial_guess

from data import constants as cst

# ID for file storing
ID = np.random.randint(1, 1e9)

# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')

# NEA departure date (mjd2000)
nea_dpt_date = int(sys.argv[1])

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
# -----------
# Number of segments
n_seg = 30
udp = Earth2NEA(nea=ast, n_seg=n_seg, tf=(arr_low, arr_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max, earth_grv=True)
problem = pg.problem(udp)

# 7 - Population
# --------------
population = pg.population(problem, size=1)

# 8 - Optimization
# ----------------
xi = initial_guess(year_=nea_dpt_date, n_seg=n_seg)

# Set the initial decision vector
population.set_x(0, xi)

# 9 - Optimization
# ----------------
population = algorithm.evolve(population)
x = population.get_x()[0]

# 10 - Post process 
# -----------------
post_process(udp, x)

# 11 - Pickle the results
# -----------------------
print("Stored with the ID : <{}>".format(ID))

# If the folder of the day hasn't been created, we create it
if not os.path.exists('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y")):
	os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y"))
	os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/Earth_NEA/')
	os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/')

# Storage of the results
with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/Earth_NEA/' + str(ID), 'wb') as f:
	pkl.dump({'udp': udp, 'population': population}, f)
