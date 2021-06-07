import pykep as pk 
import getpass
import os

def load(kernels=None):
	""" Loads the need SPICE kernels """
	# Get the host machine name
	host_nm = os.uname()[1]

	# Get the path to the spice kernels folder
	if ('node' in host_nm or 'rainman' in host_nm or 'bigmem' in host_nm or 'pando' in host_nm):
		if 'gary' in getpass.getuser():
			kernels_path = '/scratch/dcas/yv.gary/SEMBLANET/spice_kernels/'
		else:
			kernels_path = '/scratch/students/t.semblanet/spice_kernels/'
	elif 'pc' in host_nm:
		kernels_path = '/home/cesure/t.semblanet/Desktop/Asteroid-Retrieval-Mission/data/spice_kernels/'
	else:
		kernels_path = '/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/data/spice_kernels/'

	# If no specific kernels are specified, we load the default kernels
	if kernels is None:
		kernels = ['asteroids.bsp', 'de405.bsp', 'de430.bsp', 'targets.bsp']

	# Load the kernels
	for k_ in kernels:
		file = kernels_path + k_
		pk.util.load_spice_kernel(file)