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

from mpi4py import MPI

from data import constants as cst
from scripts.udp.Earth_NEA_UDP import Earth2NEA
from scripts.utils.post_process import post_process
from scripts.utils import load_sqp, load_kernels, load_bodies


def Earth_NEA(sqp, nea_dpt_date, rank):

	# If manual run, creation of the communicator
	if rank == -1:
		# Creation of the communicator 
		comm = MPI.COMM_WORLD
		rank = comm.rank

	# - * - * - * - * - * - * - * - * - * - * - * - * 
	print("Rank <{}> : Run".format(rank), flush=True)
	# - * - * - * - * - * - * - * - * - * - * - * - * 

	# Loading the main kernels
	load_kernels.load()

	# Loading of the target asteroid
	ast = load_bodies.asteroid('2020 CD3')

	# Maximum and minimum stay time [days]
	max_stay_time = 365
	min_stay_time = 90

	# 2 - Arrival date
	arr_low = pk.epoch(nea_dpt_date - max_stay_time, 'mjd2000')
	arr_upp = pk.epoch(nea_dpt_date - min_stay_time, 'mjd2000')

	# 3 - Time of flight
	tof_low = cst.YEAR2DAY * 0.1
	tof_upp = cst.YEAR2DAY * 3.00

	# 4 - Spacecraft
	m0 = 2000
	Tmax = 0.5
	Isp = 3000

	# 5 - Velocity at infinity
	vinf_max = 2e3

	# 5 - Optimization algorithm
	algorithm = load_sqp.load(sqp)

	# 7 - Problem
	udp = Earth2NEA(nea=ast, n_seg=30, tf=(arr_low, arr_upp), \
		tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max, earth_grv=True)
	problem = pg.problem(udp)

	# 7 - Population
	# --------------
	population = pg.population(problem, size=1)

	# 8 - Starting point
	# ------------------
	# Number of iterations
	N = 100
	count = 0

	found_sol = False

	# Best decision-vector
	x_best = population.get_x()[0]

	while count < N:
		# Generation of a random decision vector
		x = population.random_decision_vector()

		# Generate random decision vector until one provides a good starting point
		while udp.get_deltaV(x) > 1000 :
			x = population.random_decision_vector()

		# Set the decision vector
		population.set_x(0, x)

		# Optimization
		population = algorithm.evolve(population)

		# Mismatch error on position [km] and velocity [km/s]
		error_pos = np.linalg.norm(udp.fitness(population.champion_x)[1:4]) * pk.AU / 1000
		error_vel = np.linalg.norm(udp.fitness(population.champion_x)[4:7]) * pk.EARTH_VELOCITY / 1000
		error_mas = udp.fitness(x)[7] * udp.sc.mass

		# Update the best decision vector found
		if (udp.get_deltaV(x) > udp.get_deltaV(x_best) and udp.get_deltaV(x) < 2000 and error_pos < 10e3 and error_vel < 0.01 and abs(error_mas) < 10):
			x_best = x 
			found_sol = True
	 
		count += 1

	# Keep the best decision vector found
	population.set_x(0, x_best)

	# 11 - Pickle the results
	if found_sol == True:

		# ID for file storing
		ID = int(round(nea_dpt_date))

		# If the folder of the day hasn't been created, we create it
		if not os.path.exists('/scratch/students/t.semblanet/results/'+ date.today().strftime("%d-%m-%Y")):
			os.mkdir('/scratch/students/t.semblanet/results/'+ date.today().strftime("%d-%m-%Y"))
			os.mkdir('/scratch/students/t.semblanet/results/'+ date.today().strftime("%d-%m-%Y") + '/Earth_NEA/')
			os.mkdir('/scratch/students/t.semblanet/results/'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/')

		res = {'udp': udp, 'population': population}
		with open('/scratch/students/t.semblanet/results/' + date.today().strftime("%d-%m-%Y") + \
			'/Earth_NEA/' + str(ID) + '_' + str(sqp), 'wb') as f:
			pkl.dump(res, f)

		# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
		print("Earth -> NEA\tRank <{}> : Finished successfully!".format(rank), flush=True)
		# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *

	else:
		# - * - * - * - * - * - * - * - * - * - * - * - * - 
		print("Rank <{}> : No solution found".format(rank))
		# - * - * - * - * - * - * - * - * - * - * - * - * - 

if __name__ == '__main__':

	sqp = str(sys.argv[1])
	nea_dpt_date = str(sys.argv[2])
	rank = -1

	Earth_NEA(sqp, nea_dpt_date, rank)