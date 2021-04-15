#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mon 22 2021 10:33:20

@author: SEMBLANET Tom

"""

import os
import sys
import pickle 
from datetime import datetime as dt

import matplotlib.pyplot as plt

from scripts.utils import load_kernels

def post_process(udp, x_best):

	# Inspect the solution
	udp.brief(x_best)

	# Plot the trajectory
	udp.plot_traj(x_best)
	plt.show()

	# Plot the thrust profil
	udp.plot_thrust(x_best)
	plt.show()

if __name__ == '__main__':

	# Load the main kernels
	load_kernels.load()

	# Path of the Pickle file containing the trajectory to analyse
	file_path = str(sys.argv[1])

	# Extraction of the file content
	with open(file_path, 'rb') as file:
		results = pickle.load(file)

	# Post process the results
	post_process(results['udp'], results['population'].get_x()[0])
