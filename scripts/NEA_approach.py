import matplotlib.pyplot as plt

import pykep as pk 
import numpy as np 

from scripts.loaders import load_sqp, load_kernels, load_bodies
from data.spk_table import NAME2SPK

# Load the default SPICE kernels
load_kernels.load()

# Load the Earth from SPICE
earth = load_bodies.planet('earth')

# Load all the asteroids of interest from SPICE
asteroids = list()
asteroids_names = [nm_ for nm_ in NAME2SPK]
for i, name in enumerate(NAME2SPK):
		ast_ = load_bodies.asteroid(name)
		asteroids.append(ast_)

# Initial and final time of interest
ti = pk.epoch_from_string('2020-01-01 00:00:00')
tf = pk.epoch_from_string('2050-12-31 23:59:59')

# Construction of the time array
time = np.linspace(float(str(ti.jd)), float(str(tf.jd)), int(1e3))

# Creation of ephemeride ndarrays
earth_r = np.ndarray(shape=(len(time), 3))
asteroids_r = np.ndarray(shape=(len(asteroids), len(time), 3))

# Creation of Earth - NEA distance array
asteroids_d = np.ndarray(shape=(len(asteroids), len(time)))

# Computation of ephemerides
for i, t_ in enumerate(time):
	epoch = pk.epoch(t_, 'jd')

	earth_r[i] = earth.eph(epoch)[0]

	for j in range(len(asteroids)):
		asteroids_r[j, i] = asteroids[j].eph(epoch)[0]
		asteroids_d[j, i] = np.linalg.norm(earth_r[i] - asteroids_r[j, i])

fig = plt.figure()
ax = fig.add_subplot(111)

for i, dist in enumerate(asteroids_d):
	if asteroids_names[i] == '2020 CD3':
		ax.plot(time, dist / pk.AU, label=asteroids_names[i])

ax.set_xlabel('Time (Julian Day)')
ax.set_ylabel('Earth - NEA Distance (AU)')
plt.legend()

plt.grid()
plt.show()






	
