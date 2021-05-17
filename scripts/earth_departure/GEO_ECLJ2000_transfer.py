import numpy as np
import pykep as pk 
import pickle

import matplotlib.pyplot as plt

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.utils import cart2sph, sph2cart
from scripts.utils import load_bodies, load_kernels



def ECLJ2000_trajectory(tau, trajectories, times):
	# Kernels loading 
	load_kernels.load()

	t0 = pk.epoch(tau, julian_date_type='mjd2000')

	earth = load_bodies.planet(name='EARTH')
	moon  = load_bodies.planet(name='MOON')

	# Construction of the Earth centered frame
	r_E_ECLP, v_E_ECLP = earth.eph(t0)
	r_E_ECLP, v_E_ECLP = np.array(r_E_ECLP), np.array(v_E_ECLP)

	r_M_ECLP, v_M_ECLP = moon.eph(t0)
	r_M_ECLP, v_M_ECLP = np.array(r_M_ECLP), np.array(v_M_ECLP)

	i = (r_M_ECLP - r_E_ECLP) / np.linalg.norm(r_M_ECLP - r_E_ECLP)
	k = np.cross(r_M_ECLP, v_M_ECLP) / np.linalg.norm(r_M_ECLP) / np.linalg.norm(v_M_ECLP)
	j = np.cross(k, i)

	# Construction of the passage matrix from the ECLPJ2000 frame to the Earth centered one
	P_J20002GEO = np.array([[i[0], j[0], k[0]], 
			                [i[1], j[1], k[1]], 
			                [i[2], j[2], k[2]]])

	# Conversion of the trajectories into the ECLJ2000 frame
	r_ar = trajectories[0]
	t_ar = times[0]

	r_out = trajectories[1]
	t_out = times[1]


	r_ar_m = np.copy(r_ar)
	r_out_m = np.copy(r_out)

	# Apogee raising
	for k in range(len(t_ar)):
		# Rotation
		r_ar[:3, k] = P_J20002GEO.dot(r_ar[:3, k])
		r_ar[3:, k] = P_J20002GEO.dot(r_ar[3:, k])

		# Shift of the Moon's states
		t = pk.epoch(tau - pk.SEC2DAY*t_ar[-(k+1)], julian_date_type='mjd2000')

		r_E_ECLP, v_E_ECLP = earth.eph(t)
		r_E_ECLP, v_E_ECLP = np.array(r_E_ECLP), np.array(v_E_ECLP)

		r_M_ECLP, v_M_ECLP = moon.eph(t)
		r_M_ECLP, v_M_ECLP = np.array(r_M_ECLP), np.array(v_M_ECLP)

		r_ar_m[:3, k] = (r_M_ECLP - r_E_ECLP) / 1000
		r_ar_m[3:, k] = (v_M_ECLP - v_E_ECLP) / 1000


	# Outter trajectory
	for k in range(len(t_out)):
		# Rotation
		r_out[:3, k] = P_J20002GEO.dot(r_out[:3, k])
		r_out[3:, k] = P_J20002GEO.dot(r_out[3:, k])

		# Shift of the Moon's states
		t = pk.epoch(tau + pk.SEC2DAY*t_out[k], julian_date_type='mjd2000')

		r_E_ECLP, v_E_ECLP = earth.eph(t)
		r_E_ECLP, v_E_ECLP = np.array(r_E_ECLP), np.array(v_E_ECLP)

		r_M_ECLP, v_M_ECLP = moon.eph(t)
		r_M_ECLP, v_M_ECLP = np.array(r_M_ECLP), np.array(v_M_ECLP)

		r_out_m[:3, k] = (r_M_ECLP - r_E_ECLP) / 1000
		r_out_m[3:, k] = (v_M_ECLP - v_E_ECLP) / 1000


	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot([0], [0], [0], 'o', markersize=7, color='black')

	ax.plot(r_ar[0], r_ar[1], r_ar[2], '-', linewidth=1, color='blue')
	ax.plot(r_ar_m[0], r_ar_m[1], r_ar_m[2], '-', linewidth=1, color='black')

	ax.plot(r_out[0], r_out[1], r_out[2], '-', linewidth=1, color='blue')
	ax.plot(r_out_m[0], r_out_m[1], r_out_m[2], '-', linewidth=1, color='black')

	plt.show()

	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_tests/1', 'wb') as f:
		pickle.dump({'r': r_ar, 't': t_ar}, f)

