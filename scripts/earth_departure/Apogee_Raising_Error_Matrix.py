import os
import sys
import numpy as np 
import pickle

# # 1 - Conversion from Pickle file to Text file
# files_list = os.listdir('/home/dcas/yv.gary/SEMBLANET/Asteroid-Retrieval-Mission/local/error_matrices')
# for file in files_list:
# 	np.savetxt(fname='/home/dcas/yv.gary/SEMBLANET/Asteroid-Retrieval-Mission/local/error_matrices/'+str(file)+'.txt')



theta_list = np.array([])
files_list = os.listdir('/home/dcas/yv.gary/SEMBLANET/Asteroid-Retrieval-Mission/local/error_matrices')

for file in files_list:
	if file != 'matrix':
		theta_list = np.append(theta_list, float(file))
theta_list = np.sort(theta_list)

print("Theta : {}".format(theta_list[2847]))
print("Eps : {}".format(np.linspace(0, np.pi, 1000)[64] * 180 / np.pi))


error_matrices = np.zeros((len(theta_list), 2000, 2))

for k, theta in enumerate(theta_list):
	with open('/home/dcas/yv.gary/SEMBLANET/Asteroid-Retrieval-Mission/local/error_matrices/'+str(theta), 'rb') as file:
		sub_matrix = pickle.load(file)

	error_matrices[k, :, 0] = sub_matrix[:, 0]
	error_matrices[k, :, 1] = sub_matrix[:, 1]

with open('/home/dcas/yv.gary/SEMBLANET/Asteroid-Retrieval-Mission/local/error_matrices/matrix', 'wb') as file:
	pickle.dump(error_matrices, file)