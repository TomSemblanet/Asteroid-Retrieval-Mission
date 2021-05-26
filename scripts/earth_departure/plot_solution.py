import sys 
import numpy as np 
import pykep as pk 
import pickle

import matplotlib.pyplot as plt 

from scripts.earth_departure.utils import kepler, cart2sph, sph2cart, R2, P_GEO2HRV, P_HRV2GEO, angle_w_Ox, plot_env_2D, plot_env_3D
from scripts.earth_departure.cr3bp import CR3BP

def thrust_coast_phases(thrust_profil):
	""" Returns a list containing the index of the trajectory on a coast phase and an other list containing the 
		index of the trajectory on a thrust phase """

	coast_phases = list()
	thrust_phases = list()

	N = len(thrust_profil[0])
	index = 0

	while index < N:

		if thrust_profil[0][index] < 1e-3:
			coast = [index]
			index += 1

			while (index < N and thrust_profil[0][index] < 1e-3):
				coast.append(index)
				index += 1

			coast_phases.append(coast)

		else:
			thrust = [index]
			index += 1

			while (index < N and thrust_profil[0][index] > 1e-3):
				thrust.append(index)
				index += 1

			thrust_phases.append(thrust)

	return coast_phases, thrust_phases

	

def synodic_plot(file_path):
	""" Plots the Earth departure trajectory in the synodic frame 

		Parameters
		----------
			file_path: string
				Path to the file containing the trajectories pickled

	"""

	# Unpickle the objects
	with open(file_path, 'rb') as f:
		results = pickle.load(f)

	coast_phases, thrust_phases = thrust_coast_phases(results['thrusts'])


	fig = plt.figure(figsize=(7, 7))
	ax = fig.gca(projection='3d')

	ax.plot(results['trajectory'][0], results['trajectory'][1], results['trajectory'][2], '-', color='C0', linewidth=1.3)

	for thrust in thrust_phases:
		ax.plot(results['trajectory'][0, thrust], results['trajectory'][1, thrust], results['trajectory'][2, thrust], '-', color='red', linewidth=1.3)

	ax.plot([-0.012151], [0], [0], 'o', color='black', markersize=5)
	ax.plot([1-0.012151], [0], [0], 'o', color='black', markersize=2)

	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)
	ax.set_zlim(-1, 1)

	plt.show()


def eci_plot(file_path):
	""" Plots the Earth departure trajectory in the ECI frame 

		Parameters
		----------
			file_path: string
				Path to the file containing the trajectories pickled

	"""

	cr3bp = CR3BP(mu=0.012151, L=384400, V=384400/(2360591.424/(2*np.pi)), T=2360591.424/(2*np.pi))

	# Unpickle the objects
	with open(file_path, 'rb') as f:
		results = pickle.load(f)

	coast_phases, thrust_phases = thrust_coast_phases(results['thrusts'])

	for k, t in enumerate(results['time']):
		results['trajectory'][:, k] = cr3bp.syn2eci(t/cr3bp.T, results['trajectory'][:, k])

	fig = plt.figure(figsize=(7, 7))
	ax = fig.gca(projection='3d')

	ax.plot(results['trajectory'][0]*cr3bp.L, results['trajectory'][1]*cr3bp.L, results['trajectory'][2]*cr3bp.L, '-', color='blue', linewidth=1)

	for thrust in thrust_phases:
		ax.plot(results['trajectory'][0, thrust]*cr3bp.L, results['trajectory'][1, thrust]*cr3bp.L, results['trajectory'][2, thrust]*cr3bp.L, '-', color='red', linewidth=1)
	
	ax.set_xlim(-cr3bp.L, cr3bp.L)
	ax.set_ylim(-cr3bp.L, cr3bp.L)
	ax.set_zlim(-cr3bp.L, cr3bp.L)

	plot_env_3D(ax)

	plt.show()

if __name__ == '__main__':

	synodic_plot(sys.argv[1])





