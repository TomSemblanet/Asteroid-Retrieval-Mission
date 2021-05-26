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
from scripts.utils.double_segments import double_segments_NEA_Earth, double_segments_Earth_NEA

from data import constants as cst

""" 

This script runs the optimization of a transfer between a NEA and the Earth using low-thrust propulsion 
using ISAE-SUPAERO super-computers Rainman or Pando. 
It allows to double the number of segments of a solution to enhance its physical accuracy.

Three arguments must be provided to the script when it's runned : 
-----------------------------------------------------------------

	1) The path of the folder where the file(s) is/are stored on the supercomputer

"""

# Path to the Pickle file
file_path = sys.argv[1]

# Maximal delta-V
dV_max = float(sys.argv[2]) 

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
	print("Error.", flush=True)

population = algorithm.evolve(population)

# Recovery of the results
x = population.get_x()[0]

if udp.get_deltaV(x) >= dV_max:
	print("Delta V too high.", flush=True)

else:
	if 'NEA_Earth' in file_path:
		udp, population = double_segments_NEA_Earth(udp_=udp, population_=population)
	elif 'Earth_NEA' in file_path:
		udp, population = double_segments_Earth_NEA(udp_=udp, population_=population)

	population = algorithm.evolve(population)

	# Recovery of the results
	x = population.get_x()[0]

	if udp.get_deltaV(x) < dV_max:
		# Storage
		if 'NEA_Earth' in file_path:
			save(host='pando', mission='NEA_Earth', udp=udp, population=population, additional_sign='_doubled')
		elif 'Earth_NEA' in file_path:
			save(host='pando', mission='Earth_NEA', udp=udp, population=population, additional_sign='_doubled')
	
	else:
		print("Delta V too high.", flush=True)