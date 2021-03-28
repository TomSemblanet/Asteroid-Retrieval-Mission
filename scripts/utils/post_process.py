#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mon 22 2021 10:33:20

@author: SEMBLANET Tom

"""

import os
import pickle 
from datetime import datetime as dt

import matplotlib.pyplot as plt

def post_process(udp, x_best):

	# If we are on RAINMAIN, we pickle the results to inspect them further
	if 'node' in os.uname()[1]:
		rs = {'udp': udp, 'x': x_best}

		date = dt.now().strftime("%d_%m_%Y_%H_%M_%S")
		with open('Earth_NEA_' + str(date), 'wb') as file:
			pickle.dump(obj=rs, file=file, protocol=4)

	else:
		# 9 - Inspect the solution
		udp.report(x_best)

		# 10 - Plot the trajectory
		udp.plot_traj(x_best)
		plt.show()

		# 11 - Plot the thrust profil
		udp.plot_thrust(x_best)
		plt.show()