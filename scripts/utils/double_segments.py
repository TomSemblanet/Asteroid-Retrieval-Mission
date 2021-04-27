import os
import sys
import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl
from datetime import date

from scripts.udp.NEA_Earth.NEA_Earth_UDP import NEA2Earth
from scripts.udp.Earth_NEA.Earth_NEA_UDP import Earth2NEA
from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.utils.post_process import post_process
from scripts.utils.pickle_results import save

from data import constants as cst

def double_segments_NEA_Earth(udp_=None, population_=None):
	""" Doubles the number of segments of a trajectory between an asteroid and the Earth.

		It's important to run a second optimization once the number of nodes as been doubled so 
		the changes in the dynamic can be corrected. """

	# Loading the main kernels
	load_kernels.load()

	# Extraction of the decision vector
	x_ = population_.get_x()[0]
	throttles_ = np.array([x_[6 + 3 * i: 9 + 3 * i] for i in range(udp_.n_seg)])

	# Construction of a new UDP
	udp = NEA2Earth(nea=udp_.nea, n_seg=2 * udp_.n_seg, t0=(udp_.t0[0], udp_.t0[1]), tof=(udp_.tof[0], udp_.tof[1]), m0=udp_.sc.mass, \
					  Tmax=udp_.sc.thrust, Isp=udp_.sc.isp, nea_mass=udp_.nea_mass, vinf_max=udp_.vinf_max, earth_grv=udp_.earth_grv)
	problem = pg.problem(udp)

	# Construction of the new decision vector
	x = np.concatenate((x_[:6], np.repeat(a=throttles_, repeats=2, axis=0).flatten()))

	# Construction of the new population
	population = pg.population(problem, size=1)
	population.set_x(0, x)

	return udp, population

def double_segments_Earth_NEA(udp_=None, population_=None):
	""" Doubles the number of segments of a trajectory between the Earth and an asteroid.

		It's important to run a second optimization once the number of nodes as been doubled so 
		the changes in the dynamic can be corrected. """

	# Loading the main kernels
	load_kernels.load()

	# Extraction of the decision vector
	x_ = population_.get_x()[0]
	throttles_ = np.array([x_[6 + 3 * i: 9 + 3 * i] for i in range(udp_.n_seg)])

	# Construction of a new UDP
	udp = Earth2NEA(nea=udp_.nea, n_seg=2 * udp_.n_seg, tf=(udp_.tf[0], udp_.tf[1]), tof=(udp_.tof[0], udp_.tof[1]), m0=udp_.sc.mass, \
					  Tmax=udp_.sc.thrust, Isp=udp_.sc.isp, vinf_max=udp_.vinf_max, earth_grv=udp_.earth_grv)
	problem = pg.problem(udp)

	# Construction of the new decision vector
	x = np.concatenate((x_[:6], np.repeat(a=throttles_, repeats=2, axis=0).flatten()))

	# Construction of the new population
	population = pg.population(problem, size=1)
	population.set_x(0, x)

	return udp, population

if __name__ == '__main__':

	# Path to the Pickle file
	file_path = sys.argv[1]

	# Extraction of the Pickle file
	with open(file_path, 'rb') as file:
		data = pkl.load(file)

	# Algorithm used to correct the dynamic changes
	algorithm = load_sqp.load('ipopt')

	# First double
	if 'NEA_Earth' in file_path:
		udp, population = double_segments_NEA_Earth(udp_=data['udp'], population_=data['population'])
	elif 'Earth_NEA' in file_path:
		udp, population = double_segments_Earth_NEA(udp_=data['udp'], population_=data['population'])
	else:
		print("Error.")
		sys.exit()

	population = algorithm.evolve(population)

	# Recovery of the results
	x = population.get_x()[0]

	post_process(udp, x)

	# Storage
	save(host='laptop', mission='NEA_Earth', udp=udp, population=population)

