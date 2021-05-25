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

""" 

This script runs the optimization of a transfer between a NEA and the Earth using low-thrust propulsion 
using ISAE-SUPAERO super-computers Rainman or Pando. 
It allows to double the number of segments of a solution to enhance its physical accuracy.

Three arguments must be provided to the script when it's runned : 
-----------------------------------------------------------------

	1) The path of the folder where the file(s) is/are stored on the supercomputer

"""

# Creation of the communicator 
comm = MPI.COMM_WORLD
rank = comm.rank

# Path to the Pickle file
folder_path = sys.argv[1]

# Maximal delta-V
dV_max = int(sys.argv[2])

# List of files in the folder
files = os.listdir(folder_path)

# - * - * - * - * - * - * - * - * - * - * - * - * 
print("Rank <{}> : Run".format(rank), flush=True)
# - * - * - * - * - * - * - * - * - * - * - * - * 

# Extraction of the Pickle file
with open('/'.join([folder_path, files[min(rank, len(files)-1)]]), 'rb') as file:
	data = pkl.load(file)
	print("<{}> Opened file : {}".format(rank, '/'.join([folder_path, files[min(rank, len(files)-1)]])), flush=True)

# Algorithm used to correct the dynamic changes
algorithm = load_sqp.load('ipopt')

# First double
if 'NEA_Earth' in folder_path:
	udp, population = double_segments_NEA_Earth(udp_=data['udp'], population_=data['population'])
	print("NEA -> Earth trajectory", flush=True)
elif 'Earth_NEA' in folder_path:
	udp, population = double_segments_Earth_NEA(udp_=data['udp'], population_=data['population'])
	print("Earth -> NEA trajectory", flush=True)
else:
	print("Error.", flush=True)

# - * - * - * - * - * - * - * - * - 
print("<{}> 1st phase".format(rank), flush=True)
# - * - * - * - * - * - * - * - * - 

population = algorithm.evolve(population)

# Recovery of the results
x = population.get_x()[0]

if udp.get_deltaV(x) >= dV_max:
	# - * - * - * - * - * - * 
	print("Delta V too high.", flush=True)
	# - * - * - * - * - * - * 

else:
	# - * - * - * - * - * - * - * - * - 
	print("<{}> 2nd phase".format(rank), flush=True)
	# - * - * - * - * - * - * - * - * - 
	if 'NEA_Earth' in folder_path:
		udp, population = double_segments_NEA_Earth(udp_=udp, population_=population)
	elif 'Earth_NEA' in folder_path:
		udp, population = double_segments_Earth_NEA(udp_=udp, population_=population)

	population = algorithm.evolve(population)

	# Recovery of the results
	x = population.get_x()[0]

	if udp.get_deltaV(x) < dV_max:
		# Storage
		if 'NEA_Earth' in folder_path:
			save(host='pando', mission='NEA_Earth', udp=udp, population=population, additional_sign='_doubled')
		elif 'Earth_NEA' in folder_path:
			save(host='pando', mission='Earth_NEA', udp=udp, population=population, additional_sign='_doubled')
	
	else:
		# - * - * - * - * - * - * 
		print("Delta V too high.", flush=True)
		# - * - * - * - * - * - * 