import numpy as np 
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt 

from scripts.earth_departure.cr3bp import CR3BP

# Unpickle the DROs informations
with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/data/DROs', 'rb') as file:
	DROs = pickle.load(file)

# colormap
cmap = plt.get_cmap('jet', 100)


mu = 0.012151
L = 384400
T = 2360591.424
cr3bp = CR3BP(mu=mu, L=L, V=L/(T/(2*np.pi)), T=T/(2*np.pi))


fig = plt.figure()
ax = fig.add_subplot(111)

for k, Cjac in enumerate(DROs.keys()):
	if 	k % 10 == 0:
		ax.plot(DROs[Cjac][:, 0], DROs[Cjac][:, 1], '-', color=cmap(float(Cjac)-2), linewidth=1)

# Normalizer
norm = mpl.colors.Normalize(vmin=2, vmax=3)

# creating ScalarMappable
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

plt.colorbar(sm)

ax.plot([1-0.012151], [0], 'o', markersize=2, color='black')
ax.plot([-0.012151], [0], 'o', markersize=5, color='black')

ax.set_xlabel('X [-]')
ax.set_ylabel('Y [-]')

ax.set_title('Distant Retrograde Orbits')

plt.grid()
plt.show()
