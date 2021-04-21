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

from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.udp.NEA_Earth_UDP import NEA2Earth
from scripts.utils.post_process import post_process
from data import constants as cst


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
lw_low = pk.epoch_from_string('2021-01-01 00:00:00')
lw_upp = pk.epoch_from_string('2050-12-31 23:59:59')

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

# 6 - Problem
# -----------
udp = NEA2Earth(nea=ast, n_seg=30, t0=(lw_low, lw_upp), tof=(tof_low, tof_upp), m0=m0, \
	Tmax=Tmax, Isp=Isp, nea_mass=ast_mass, phi_min=phi_min, phi_max=phi_max, theta_min=theta_min, \
	theta_max=theta_max, earth_grv=True)
problem = pg.problem(udp)

# 7 - Population
# --------------
population = pg.population(problem, size=1)

# Update the best result
population.set_x(0, x)

# Inspect the solution
post_process(udp, population.get_x()[0])

# ID for file storing
nea_dpt_date = pk.epoch(x[0]).mjd2000
ID = int(round(float((nea_dpt_date)), 0))


# If the folder of the day hasn't been created, we create it
if not os.path.exists('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y")):
	os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y"))
	os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/Earth_NEA/')
	os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/')

# Storage of the results
with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/' + str(ID), 'wb') as f:
	pkl.dump({'udp': udp, 'population': population}, f)



