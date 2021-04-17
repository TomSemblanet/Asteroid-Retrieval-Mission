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

# Define a random ID for the results storage
ID = np.random.randint(0, 1e9)

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

# 5 - Optimization algorithm
# --------------------------
algorithm = load_sqp.load('ipopt')

# 6 - Problem
# -----------
udp = NEA2Earth(nea=ast, n_seg=20, t0=(lw_low, lw_upp), tof=(tof_low, tof_upp), m0=m0, \
	Tmax=Tmax, Isp=Isp, nea_mass=ast_mass, phi_min=phi_min, phi_max=phi_max, theta_min=theta_min, \
	theta_max=theta_max, earth_grv=True)
problem = pg.problem(udp)

# 7 - Population
# --------------
population = pg.population(problem, size=1)

# 8 - Starting point
# ------------------
# Number of iterations
N = 20
count = 0

found_sol = False

# Best decision-vector
x_best = population.get_x()[0]

while count < N:
	# Generation of a random decision vector
	x = population.random_decision_vector()

	# Generate random decision vector until one provides a good starting point
	while udp.get_deltaV(x) > 1000:
		x = population.random_decision_vector()

	# Set the decision vector
	population.set_x(0, x)

	# Optimization
	population = algorithm.evolve(population)
	x = population.get_x()[0]

	# Mismatch error on position [km] and velocity [km/s]
	error_pos = np.linalg.norm(udp.fitness(x)[1:4]) * pk.AU / 1000
	error_vel = np.linalg.norm(udp.fitness(x)[4:7]) * pk.EARTH_VELOCITY / 1000
	error_mas = np.linalg.norm(udp.fitness(x)[7]) * udp.sc.mass

	# Update the best decision vector found
	if (udp.get_deltaV(x) < udp.get_deltaV(x_best) and udp.get_deltaV(x) < 500 and error_pos < 100e3 and error_vel < 0.05 and abs(error_mas) < 10):
		x_best = x
		found_sol = True

	count += 1

# Update the best result
population.set_x(0, x_best)


# 12 - Pickle the results
if found_sol == True:

	# ID for file storing
	nea_dpt_date = pk.epoch(x_best[0]).mjd2000
	ID = int(round(float((nea_dpt_date)), 0))

	print("Acceptable solution found!\nStored with the ID : <{}>".format(ID))
	input()

	# If the folder of the day hasn't been created, we create it
	if not os.path.exists('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y")):
		os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y"))
		os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/Earth_NEA/')
		os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/')

	# Storage of the results
	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/' + str(ID), 'wb') as f:
		pkl.dump({'udp': udp, 'population': population}, f)


	# Launch the Earth -> NEA scripts
	os.system("python -m scripts.missions.Earth_NEA " + str(nea_dpt_date))

else:
	print("Failure.")


