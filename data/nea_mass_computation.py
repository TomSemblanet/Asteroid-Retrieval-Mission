import numpy as np 

def get_mass(H, a=0.5):
	""" Computation of the mass of an asteroid given its absolute magnitude (H)
		and its albedo (a) """

	# Computation of the asteroid diameter [m]
	D = 1000 * 1329 / np.sqrt(a) * 10 ** (- 0.2 * H)

	# Computation of the asteroids mass [kg] assuming a mean volumic mass
	# rho = 2600 kg/m^3 and a spherical shape for the body
	m = 4 / 3 * np.pi * (0.5 * D)**3 * 2600

	return m
