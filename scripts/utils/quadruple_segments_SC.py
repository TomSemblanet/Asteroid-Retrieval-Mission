import os
import sys
import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl
from datetime import date

from mpi4py import MPI

from scripts.udp.NEA_Earth.NEA_Earth_UDP import NEA2Earth
from scripts.udp.Earth_NEA.Earth_NEA_UDP import Earth2NEA
from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.utils.post_process import post_process
from scripts.utils.pickle_results import save
from scripts.utils.double_segments import double_segments_NEA_Earth, double_segments_Earth_NEA

from data import constants as cst

# Creation of the communicator 
comm = MPI.COMM_WORLD
rank = comm.rank

# Path to the Pickle file
folder_path = sys.argv[1]

# List of files in the folder
files = os.listdir(folder_path)

# - * - * - * - * - * - * - * - * - * - * - * - * 
print("Rank <{}> : Run".format(rank), flush=True)
# - * - * - * - * - * - * - * - * - * - * - * - * 

# Extraction of the Pickle file
with open('/'.join([folder_path, files[max(rank, len(files)-1)]]), 'rb') as file:
	data = pkl.load(file)

# Algorithm used to correct the dynamic changes
algorithm = load_sqp.load('ipopt')

# First double
if 'NEA_Earth' in folder_path:
	udp, population = double_segments_NEA_Earth(udp_=data['udp'], population_=data['population'])
elif 'Earth_NEA' in folder_path:
	udp, population = double_segments_Earth_NEA(udp_=data['udp'], population_=data['population'])
else:
	print("Error.")
	sys.exit()

population = algorithm.evolve(population)

# Recovery of the results
x = population.get_x()[0]

if udp.get_deltaV(x) > 300:
	# - * - * - * - * - * - * 
	print("Delta V too high.", flush=True)
	# - * - * - * - * - * - * 
	sys.exit()

else:
	udp, population = double_segments_NEA_Earth(udp_=udp, population_=population)

	population = algorithm.evolve(population)

	# Recovery of the results
	x = population.get_x()[0]

	if udp.get_deltaV < 300:
		# Storage
		if 'NEA_Earth' in folder_path:
			save(host='pando', mission='NEA_Earth', udp=udp, population=population)
		elif 'Earth_NEA' in folder_path:
			save(host='pando', mission='Earth_NEA', udp=udp, population=population)
	
	else:
		# - * - * - * - * - * - * 
		print("Delta V too high.", flush=True)
		# - * - * - * - * - * - * 