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

	# Inspect the solution
	udp.report(x_best)

	# Plot the trajectory
	udp.plot_traj(x_best)
	plt.show()

	# Plot the thrust profil
	udp.plot_thrust(x_best)
	plt.show()