import sys

import pickle as pkl
import pygmo as pg 
import pykep as pk 
import numpy as np 

from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.utils.post_process import post_process

def mbh_(file_path=None):
	""" Runs an optimization of a NEA -> Earth trajectory using the Monotonic-Bassin-Hopping 
		algorithm. Briefly, this algorithm explores the environment of the solution to enhance
		the results. """

	# Loading the main kernels
	load_kernels.load()

	# Path to the Pickle file
	file_path = sys.argv[1]

	# Extraction of the Pickle file
	with open(file_path, 'rb') as file:
		data = pkl.load(file)

	# Algorithm used to optimize the solution
	intern_algo = load_sqp.load('ipopt')

	# User-Defined Problem (udp) and population
	udp = data['udp']
	population = data['population']

	# Monotonic-Basin-Hopping algorithmn instantiation
	mbh = pg.algorithm(pg.mbh(algo=intern_algo, stop=5))
	mbh.set_verbosity(3)

	# Optimization
	population = mbh.evolve(population)

	# Post-process
	post_process(udp, population.get_x()[0])
	udp.brief(population.get_x()[0])


if __name__ == '__main__':
	
	mbh_(sys.argv[1])



