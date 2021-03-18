import pykep as pk 
import os

def load_kernels(kernels=None):
	""" Loads the need SPICE kernels """
	# Get the host machine name
	host_nm = os.uname()[1]

	# Get the path to the spice kernels folder
	if 'node' in host_nm:
		kernels_path = '/scratch/students/t.semblanet/spice_kernels/'
	else:
		kernels_path = '/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/data/spice_kernels/'

	# If no specific kernels are specified, we load the default kernels
	if kernels is None:
		kernels = ['asteroids.bsp', 'de405.bsp', 'de430.bsp']

	# Load the kernels
	for k_ in kernels:
		file = kernels_path + k_
		pk.util.load_spice_kernel(file)