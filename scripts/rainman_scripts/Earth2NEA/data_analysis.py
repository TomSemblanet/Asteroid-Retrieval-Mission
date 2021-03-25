#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 2021 10:36:00

@author: SEMBLANET Tom

"""

import os
import sys
import pykep as pk 
import pickle as pkl
import matplotlib.pyplot as plt

from scripts.loaders import load_kernels

def meta_load_file(dir_):
	""" Load all the Pickle files containing the results of many optimization runned on
		the ISAE-SUPAEOR super-computer RAINMAN 

		Parameters:
		-----------
		dir_ : string
			Directory where the files are stored

	"""

	# List of Pickle files
	files_lst = os.listdir(dir_)

	for file in files_lst:
		print("Loading file {}".format('/'.join([dir_, file])))

		with open('/'.join([dir_, file]), 'rb') as fl:
			res = pkl.load(fl)
		input()

def load_file(name, id_):
	""" Load a Pickle file containing the results of an optimisation runned on the
		ISAE-SUPAERO super-computer RAINMAN

		Parameters:
		-----------
		name: string
			Name of the mission  (eg. Earth_NEA)
		id : string
			ID of the mission (eg. 250)

		Returns:
		--------
		udp: <pykep.udp>
			Original UDP
		dv: array
			Optimized decision vector

	"""
	# Construct the file name
	file_name = '/'.join(['/scratch/students/t.semblanet', name + '_results', id_])

	print(os.listdir(file_name))
	input()

	with open(file_name, 'rb') as file:
		res = pkl.load(file)

	return res['udp'], res['x']

def plot_trajectory(udp, dv):
	""" Plot the a trajectory resulting of an optimization on the 
		ISAE-SUPAERO super-computer RAINMAN 

		Parameters:
		-----------
		udp: <pykep.udp>
			Original UDP containing informations necessary to the trajectory plot
		dv: array
			Optimized decision vector

	"""
	udp.plot_traj(dv)
	plt.show()

def plot_thrust(udp, dv):
	""" Plot the the thrust over a trajectory resulting of an optimization on the 
		ISAE-SUPAERO super-computer RAINMAN 

		Parameters:
		-----------
		udp: <pykep.udp>
			Original UDP containing informations necessary to the trajectory plot
		dv: array
			Optimized decision vector

	"""
	udp.plot_dists_thrust(dv)

if __name__ == '__main__':
	# Load the SPICE Kernels files 
	load_kernels.load()

	# Get the mission name (eg. 'Earth_NEA') and the ID (eg. 250)
	mission_nm = str(sys.argv[1])
	mission_id = str(sys.argv[2])

	dir_ = '/'.join(['/scratch/students/t.semblanet', mission_nm + '_results', mission_id])

	meta_load_file(dir_)

	# Get the optimization results
	# udp, dv = load_file(mission_nm, mission_id)



	# # Plot the trajectory
	# plot_trajectory(udp, dv)

	# # Plot the thrust profil
	# plot_thrust(udp, dv)