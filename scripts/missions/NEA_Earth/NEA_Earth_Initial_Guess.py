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

from data.nea_mass_computation import get_mass

from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.utils.post_process import post_process

from scripts.udp.NEA_Earth.NEA_Earth_position_UDP import NEA2Earth_Pos
from scripts.udp.NEA_Earth.NEA_Earth_velocity_UDP import NEA2Earth_Vel

from data import constants as cst

def initial_guess(year_, n_seg):
	# Define a random ID for the results storage
	ID = np.random.randint(0, 1e9)

	# Year of interest
	year = year_

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
	tof_upp = cst.YEAR2DAY * 4.00

	# 4 - Spacecraft
	# --------------
	m0 = 10000 + ast_mass
	Tmax = 2
	Isp = 3000

	# 5 - Earth arrival 
	# -----------------
	vinf_max = 3.5e3

	# 5 - Optimization algorithm
	# --------------------------
	algorithm = load_sqp.load('ipopt', max_iter=300)

	# 6 - Optimization on position
	# ----------------------------
	udp = NEA2Earth_Pos(nea=ast, n_seg=n_seg, t0=(lw_low, lw_upp), tof=(tof_low, tof_upp), m0=m0, \
		Tmax=Tmax, Isp=Isp, nea_mass=ast_mass, vinf_max=vinf_max, earth_grv=True)
	problem = pg.problem(udp)

	population = pg.population(problem, size=1)

	x = population.random_decision_vector()

	# Set the decision vector
	population.set_x(0, x)

	# Optimization
	# * - * - * - * - * - * - * - * - * - 
	print("Optimization on the position", flush=True)
	# * - * - * - * - * - * - * - * - * - 
	population = algorithm.evolve(population)
	x = population.get_x()[0]


	# 7 - Optimization on velocity
	# ----------------------------
	udp = NEA2Earth_Vel(nea=ast, n_seg=n_seg, t0=(lw_low, lw_upp), tof=(tof_low, tof_upp), m0=m0, \
		Tmax=Tmax, Isp=Isp, nea_mass=ast_mass, vinf_max=vinf_max, earth_grv=True)
	problem = pg.problem(udp)

	population = pg.population(problem, size=1)

	# Set the decision vector
	population.set_x(0, x)

	# Optimization
	# * - * - * - * - * - * - * - * - * - 
	print("Optimization on the velocity", flush=True)
	# * - * - * - * - * - * - * - * - * - 
	population = algorithm.evolve(population)
	x = population.get_x()[0]

	return x



