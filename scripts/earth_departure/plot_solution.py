import sys 
import numpy as np 
import pykep as pk 
import pickle

import matplotlib.pyplot as plt 

from scripts.earth_departure.utils import kepler, cart2sph, sph2cart, R2, P_GEO2HRV, P_HRV2GEO, angle_w_Ox, plot_env_2D, plot_env_3D

def thrust_coast_phases(thrust_profil):
	""" Returns a list containing the index of the trajectory on a coast phase and an other list containing the 
		index of the trajectory on a thrust phase """

	coast_ind = np.empty(0)
	thrust_ind = np.empty(0)

	for k, T in enumerate(thrust_profil[0]):
		if T < 1e-2:
			coast_ind = np.append(coast_ind, k)
		else:
			thrust_ind = np.append(thrust_ind, k)

	return coast_ind, thrust_ind

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

	# Extraction of the apogee raising data
	apogee_raising_traj, apogee_raising_time, apogee_raising_cr3bp, apogee_raising_thrust = \
				results['apogee_raising']['trajectory'], results['apogee_raising']['time'], results['apogee_raising']['cr3bp'], results['apogee_raising']['thrusts']
	# Extraction of the moon-moon leg data
	moon_moon_traj, moon_moon_time, moon_moon_cr3bp, moon_moon_thrusts = \
				results['moon_moon']['trajectory'], results['moon_moon']['time'], results['moon_moon']['cr3bp'], results['moon_moon']['thrusts']

	# Separation of the coast and thrust phase on the apogee raising trajectory
	apogee_raising_c_index, apogee_raising_t_index = thrust_coast_phases(apogee_raising_thrust)
	apogee_raising_c_traj = apogee_raising_traj[:, apogee_raising_c_index]
	apogee_raising_t_traj = apogee_raising_traj[:, apogee_raising_t_index]

	# Separation of the coast and thrust phase on the moon-moon trajectory
	moon_moon_c_phase, moon_moon_t_phase = thrust_coast_phases(moon_moon_thrust)
	moon_moon_c_traj = moon_moon_traj[:, moon_moon_c_index]
	moon_moon_t_traj = moon_moon_traj[:, moon_moon_t_index]

	






