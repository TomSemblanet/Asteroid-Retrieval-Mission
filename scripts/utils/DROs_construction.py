import os
import sys
import pickle
import numpy as np 


# DROs dictionnary (keys are the Jacobian constant of the orbit)
DROs = dict()

# Path of the folder where text files are stored
folder_path = sys.argv[1]

for file in os.listdir(folder_path)[1:]:
	print(file)

	# Path of the file containing the DRO states
	file_path = '/'.join([folder_path, file])

	# Conversion of the file to a matrix
	dro = np.loadtxt(file_path)

	# Jacobian constant 
	Cjac = file_path.split('/')[-1]

	DROs[Cjac] = dro


# Pickle of the results
with open('/'.join(['/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/data/DROs']), 'wb') as file:
	pickle.dump(DROs, file)

