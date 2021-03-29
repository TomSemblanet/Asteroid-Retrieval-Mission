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

from scripts.utils.loaders import load_kernels

def meta_load_file(host, dir_):
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

		# Extraction of the UDP and the population from the Pickle object
		udp, population = res['udp'], res['population']

		analysis(host, udp, population.champion_x, file[-4:])

def analysis(host, udp, dv, year):
	""" Analyses a trajectory for a given mission, save the Trajectory plot, the thrust
		profile and the main informations about the trajectory

		Parameters:
		-----------
		host: str
			Name of the supercomputer (eg. pando)
		udp: <pykep.udp>
			Original UDP containing informations necessary to the trajectory plot
		dv: array
			Optimized decision vector
		year: string
			Launch year

	"""

	# Save the 3D trajectory of the spacecraft
	fig, ax = udp.plot_traj(dv)

	fname_plot = '/'.join(['/home/cesure/t.semblanet/Desktop/Asteroid-Retrieval-Mission/' + host, mission_nm + \
		'_results', mission_id + '_data', year + 'traj'])
	fig.savefig(fname=fname_plot)

	# Save the thrust profil of the spacecraft
	fig, ax = udp.plot_thrust(dv)

	fname_thrust = '/'.join(['/home/cesure/t.semblanet/Desktop/Asteroid-Retrieval-Mission/' + host, mission_nm + \
		'_results', mission_id + '_data', year + 'thrust'])
	fig.savefig(fname=fname_thrust)

	# Save the trajectory numerical characteristic into a text file
	f_traj = open('/'.join(['/home/cesure/t.semblanet/Desktop/Asteroid-Retrieval-Mission/' + host, mission_nm + \
		'_results', mission_id + '_data', year + '_transfer_data.txt']), 'a')
	data = udp.report(x=dv, print=False)
	print(data, file=f_traj)

	# Save the constraints violation data into a text file
	f_tr = open('/'.join(['/home/cesure/t.semblanet/Desktop/Asteroid-Retrieval-Mission/' + host, mission_nm + \
		'_results', mission_id + '_data', year + '_constraints_violation_data.txt']), 'a')
	con_viol = udp.check_con_violation(x=dv, print=False)
	print(con_viol, file=f_tr)

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
	udp.plot_thrust(dv)

if __name__ == '__main__':
	# Load the SPICE Kernels files 
	load_kernels.load()

	# Get the host name
	host = str(sys.argv[1])

	# Get the mission name (eg. 'Earth_NEA') and the ID (eg. 250)
	mission_nm = str(sys.argv[2])
	mission_id = str(sys.argv[3])

	dir_ = '/'.join(['/home/cesure/t.semblanet/Desktop/Asteroid-Retrieval-Mission/supercomputers/' + host, mission_nm + '_results', mission_id])

	meta_load_file(host, dir_)