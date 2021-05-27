import sys
import numpy as np 
import pykep as pk
import pickle

import matplotlib.pyplot as plt

from scripts.utils import load_bodies, load_kernels

def get(file_path):
	""" Returns the S/C position and velocity relatively to the Earth and the Moon at the Moon arrival """

	file_path = sys.argv[1]

	with open(file_path, 'rb') as file:
		results = pickle.load(file)

	udp = results['udp']
	x = results['population'].get_x()[0]

	t0, tof = x[:2]

	load_kernels.load()

	t0 = pk.epoch(t0 + tof, julian_date_type='mjd2000')

	earth = load_bodies.planet(name='EARTH')
	moon  = load_bodies.planet(name='MOON')

	r_E, v_E = earth.eph(t0)
	r_E, v_E = np.array(r_E), np.array(v_E)

	r_M, v_M = moon.eph(t0)
	r_M, v_M = np.array(r_M), np.array(v_M)

	_, rbwd, _, vbwd, _, _, _, _, _, _, _, _ = udp.propagate(x)

	r_f = rbwd[-1]
	v_f = vbwd[-1]

	return np.concatenate((r_f, v_f)), np.concatenate((r_f - r_E, v_f - v_E)), np.concatenate((r_f - r_M, v_f - v_M))