import os
import sys
import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl
from datetime import date

from scripts.udp.NEA_Earth_UDP import NEA2Earth
from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.utils.post_process import post_process

from data import constants as cst

def double_segments_NEA_Earth(udp_=None, population_=None):
	""" Doubles the number of segments of a trajectory between an asteroid and the Earth.

		It's important to run a second optimization once the number of nodes as been doubled so 
		the changes in the dynamic can be corrected. """

	# Loading the main kernels
	load_kernels.load()

	# Extraction of the decision vector
	x_ = population_.get_x()[0]
	throttles_ = np.array([x_[5 + 3 * i: 8 + 3 * i] for i in range(udp_.n_seg)])

	# Construction of a new UDP
	udp = NEA2Earth(nea=udp_.nea, n_seg=2 * udp_.n_seg, t0=(udp_.t0[0], udp_.t0[1]), tof=(udp_.tof[0], udp_.tof[1]), m0=udp_.sc.mass, \
					  Tmax=udp_.sc.thrust, Isp=udp_.sc.isp, nea_mass=udp_.nea_mass, phi_min=udp_.phi_min, phi_max=udp_.phi_max, theta_min=udp_.theta_min, \
					  theta_max=udp_.theta_max, earth_grv=udp_.earth_grv)
	problem = pg.problem(udp)

	# Construction of the new decision vector
	x = np.concatenate((x_[:5], np.repeat(a=throttles_, repeats=2, axis=0).flatten()))

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
	udp, population = double_segments_NEA_Earth(udp_=data['udp'], population_=data['population'])
	population = algorithm.evolve(population)

	# Recovery of the results
	x = population.get_x()[0]

	post_process(udp, x)
	udp.brief(x)

	# ID for file storing
	nea_dpt_date = pk.epoch(x[0]).mjd2000
	ID = int(round(float((nea_dpt_date)), 0))

	# If the folder of the day hasn't been created, we create it
	if not os.path.exists('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y")):
		os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y"))
		os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/Earth_NEA/')
		os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/')

	# Storage of the results
	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/' + str(ID) + '_d', 'wb') as f:
		pkl.dump({'udp': udp, 'population': population}, f)

