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

from scripts.missions.NEA_Earth_Initial_Guess import initial_guess

from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.utils.post_process import post_process
from scripts.utils.mbh import mbh

from data import constants as cst



# def generate_initial_guess(x):

# 	# New decision vector
# 	# - * - * - * - * - *

# 	x_ = np.zeros(len(x))

# 	# Departure date
# 	x_[0] = x[0]

# 	# Time of flight
# 	while x[1] < 750:
# 		x[1] = np.random.random() * (1500 - 750) + 750
# 	x_[1] = x[1]

# 	# Final mass
# 	x_[2] = 0.99 * m0

# 	# Phi / Theta
# 	x_[3] = x[3]
# 	x_[4] = x[4]

# 	# - * - * - * - * - * -
# 	# Throttles managements
# 	# - * - * - * - * - * - 

# 	# Construction of the Earth object
# 	earth = load_bodies.planet('earth')

# 	# Construction of the Earth centered coordinate system
# 	# ----------------------------------------------------
# 	r_e, v_e = earth.eph(pk.epoch(x[0] + x[1]))
# 	v_e_unt = v_e / np.linalg.norm(v_e)

# 	i_unt_e = (v_e / np.linalg.norm(v_e))
# 	k_unt_e = np.cross(r_e / np.linalg.norm(r_e), i_unt_e)
# 	j_unt_e = np.cross(k_unt_e, i_unt_e)

# 	# Construction of the Asteroids centered coordinate system
# 	# --------------------------------------------------------
# 	r_ast, v_ast = ast.eph(pk.epoch(x[0]))
# 	v_ast_unt = v_ast / np.linalg.norm(v_ast)

# 	# Coordinate system
# 	i_unt_ast = (v_ast / np.linalg.norm(v_ast))
# 	k_unt_ast = np.cross(r_ast / np.linalg.norm(r_ast), i_unt_ast)
# 	j_unt_ast = np.cross(k_unt_ast, i_unt_ast)

# 	error_pos = 1e10
# 	while error_pos > 100e3:

# 		a = 10 * np.random.uniform(-1, 1)
# 		b = 180 * np.random.uniform(-1, 1)
# 		c = 90 * np.random.uniform(-1, 1)
# 		d = 180 * np.random.uniform(-1, 1)

# 		# Throttle angles in the Earth coordinate system
# 		phi_e = a * cst.DEG2RAD
# 		theta_e = b * cst.DEG2RAD

# 		# Throttle angles in the Asteroid coordinate system
# 		phi_ast = c * cst.DEG2RAD
# 		theta_ast = d * cst.DEG2RAD

# 		# Throttle vector in the Earth coordinate system
# 		# ----------------------------------------------
# 		i_e = np.cos(phi_e) * np.sin(theta_e)
# 		j_e = np.sin(phi_e) * np.sin(theta_e)
# 		k_e = np.cos(theta_e)
# 		vec_earth_basis = i_e * i_unt_e + j_e * j_unt_e + k_e * k_unt_e

# 		# Throttle vector in the Asteroid coordinate system
# 		# -------------------------------------------------
# 		i_ast = np.cos(phi_ast) * np.sin(theta_ast)
# 		j_ast = np.sin(phi_ast) * np.sin(theta_ast)
# 		k_ast = np.cos(theta_ast)
# 		vec_ast_basis = i_ast * i_unt_ast + j_ast * j_unt_ast + k_ast * k_unt_ast


# 		# Throttle vectors
# 		for i in range(2):
# 			x_[5 + 3*i: 8 + 3*i] = vec_ast_basis

# 		for i in range(5):
# 			if i==0:
# 				x_[-3:] = vec_earth_basis
# 			else:
# 				x_[-3 - 3*i: - 3*i] = vec_earth_basis

# 		post_process(udp, x_)
# 		error_pos = np.linalg.norm(udp.fitness(x_)[1:4]) * pk.AU / 1000
# 		print(error_pos)

# 	# fig = plt.figure()
# 	# ax = fig.gca(projection='3d')

# 	# # Plot the Earth orbit
# 	# pk.orbit_plots.plot_planet(earth, pk.epoch(x[0] + x[1]), units=pk.AU, legend=True, color=(0.7, 0.7, 1), axes=ax)
# 	# pk.orbit_plots.plot_planet(ast, pk.epoch(x[0]), units=pk.AU, legend=True, color=(0.7, 0.7, 1), axes=ax)
# 	# ax.plot([0], [0], [0], 'o', color='yellow')

# 	# x_ast, y_ast, z_ast = np.array(ast.eph(pk.epoch(x[0]))[0]) / pk.AU

# 	# ax.plot([x_ast, x_ast + i_unt_ast[0]/5], [y_ast + 0, y_ast + i_unt_ast[1]/5], [z_ast + 0, z_ast + i_unt_ast[2]/5])
# 	# ax.plot([x_ast, x_ast + j_unt_ast[0]/5], [y_ast + 0, y_ast + j_unt_ast[1]/5], [z_ast + 0, z_ast + j_unt_ast[2]/5])
# 	# ax.plot([x_ast, x_ast + k_unt_ast[0]/300], [y_ast + 0, y_ast + k_unt_ast[1]/300], [z_ast + 0, z_ast + k_unt_ast[2]/300])
# 	# ax.plot([x_ast, x_ast + vec_ast_basis[0]/5], [y_ast + 0, y_ast + vec_ast_basis[1]/5], [z_ast + 0, z_ast + vec_ast_basis[2]/5], linewidth=2)

# 	# x_ear, y_ear, z_ear = np.array(earth.eph(pk.epoch(x[0] + x[1]))[0]) / pk.AU

# 	# ax.plot([x_ear, x_ear + i_unt_e[0]/5], [y_ear + 0, y_ear + i_unt_e[1]/5], [z_ear + 0, z_ear + i_unt_e[2]/5])
# 	# ax.plot([x_ear, x_ear + j_unt_e[0]/5], [y_ear + 0, y_ear + j_unt_e[1]/5], [z_ear + 0, z_ear + j_unt_e[2]/5])
# 	# ax.plot([x_ear, x_ear + k_unt_e[0]/300], [y_ear + 0, y_ear + k_unt_e[1]/300], [z_ear + 0, z_ear + k_unt_e[2]/300])
# 	# ax.plot([x_ear, x_ear + vec_earth_basis[0]/5], [y_ear + 0, y_ear + vec_earth_basis[1]/5], [z_ear + 0, z_ear + vec_earth_basis[2]/5], linewidth=2)

# 	# plt.show()	

# 	return x_


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
# Define a random ID for the results storage
ID = np.random.randint(0, 1e9)
print("Stored with the ID : <{}>".format(ID))

# If the folder of the day hasn't been created, we create it
if not os.path.exists('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y")):
	os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y"))
	os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/Earth_NEA/')
	os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/')

# Storage of the results
with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/' + str(ID), 'wb') as f:
	pkl.dump({'udp': udp, 'population': population}, f)


