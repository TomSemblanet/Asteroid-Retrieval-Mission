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

from scripts.udp.Earth_NEA_position_UDP import Earth_NEA_Pos
from scripts.udp.Earth_NEA_velocity_UDP import Earth_NEA_Vel
from scripts.utils.post_process import post_process
from data import constants as cst

def initial_guess(nea_dpt_date_, n_seg):

	# Loading of the target asteroid
	ast = load_bodies.asteroid('2020 CD3')

	# NEA departure date (mjd2000)
	nea_dpt_date = nea_dpt_date_

	# 1 - Maximum and minimum stay time [days]
	# ----------------------------------------
	max_stay_time = 365
	min_stay_time = 90

	# 2 - Arrival date
	# ----------------
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

	# 5 - Velocity at infinity
	# ------------------------
	vinf_max = 2e3

	# 6 - Optimization algorithm
	# --------------------------
	algorithm = load_sqp.load('ipopt')

	# 7 - Optimization on position
	# ----------------------------
	udp = Earth_NEA_Pos(nea=ast, n_seg=n_seg, tf=(arr_low, arr_upp), \
		tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max, earth_grv=True)
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

	# 8 - Optimization on velocity
	# ----------------------------
	udp = Earth_NEA_Vel(nea=ast, n_seg=n_seg, tf=(arr_low, arr_upp), \
		tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max, earth_grv=True)
	problem = pg.problem(udp)

	population = pg.population(problem, size=1)

	x = population.random_decision_vector()

	# Set the decision vector
	population.set_x(0, x)

	# Optimization
	# * - * - * - * - * - * - * - * - * - 
	print("Optimization on the velocity", flush=True)
	# * - * - * - * - * - * - * - * - * - 
	population = algorithm.evolve(population)
	x = population.get_x()[0]

	post_process(udp, x)

	return x
