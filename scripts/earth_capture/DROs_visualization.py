import numpy as np 
import pickle

import matplotlib.pyplot as plt 

from scripts.earth_departure.cr3bp import CR3BP

# Unpickle the DROs informations
with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/data/DROs', 'rb') as file:
	DROs = pickle.load(file)


mu = 0.012151
L = 384400
T = 2360591.424
cr3bp = CR3BP(mu=mu, L=L, V=L/(T/(2*np.pi)), T=T/(2*np.pi))


fig = plt.figure()
ax = fig.add_subplot(111)

for k, Cjac in enumerate(DROs.keys()):
	if 	k % 10 == 0:
		ax.plot(DROs[Cjac][:, 0], DROs[Cjac][:, 1], '-', color='blue', linewidth=1)
		print("Jacobi constant : {}".format(cr3bp.jacobian_constant(DROs[Cjac][0, :])))

ax.plot([1-0.012151], [0], 'o', markersize=2, color='black')
ax.plot([-0.012151], [0], 'o', markersize=5, color='black')

plt.grid()
plt.show()
