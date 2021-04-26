import os
import sys
import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl
from datetime import date

def save(host, mission, udp, population):

	# Define a random ID for the results storage
	ID = np.random.randint(0, 1e9)

	if host == 'laptop':
		# If the folder of the day hasn't been created, we create it
		if not os.path.exists('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y")):
			os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y"))
			os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/Earth_NEA/')
			os.mkdir('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/')

		# Storage of the results
		with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/'+ date.today().strftime("%d-%m-%Y") + '/' + str(mission) + '/' + str(ID), 'wb') as f:
			pkl.dump({'udp': udp, 'population': population}, f)

	elif host == 'rainman':
		print(":)", flush=True)
		# If the folder of the day hasn't been created, we create it
		if not os.path.exists('/scratch/students/t.semblanet/results/'+ date.today().strftime("%d-%m-%Y")):
			os.mkdir('/scratch/students/t.semblanet/results'+ date.today().strftime("%d-%m-%Y"))
			os.mkdir('/scratch/students/t.semblanet/results'+ date.today().strftime("%d-%m-%Y") + '/Earth_NEA/')
			os.mkdir('/scratch/students/t.semblanet/results'+ date.today().strftime("%d-%m-%Y") + '/NEA_Earth/')

		# Storage of the results
		with open('/scratch/students/t.semblanet/results'+ date.today().strftime("%d-%m-%Y") + '/' + str(mission) + '/' + str(ID), 'wb') as f:
			pkl.dump({'udp': udp, 'population': population}, f)

	# - * - * - * - * - * - * - * - * - * - * - * - *
	print("Stored with the ID : <{}>".format(ID))
	# - * - * - * - * - * - * - * - * - * - * - * - *