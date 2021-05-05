import sys
import numpy as np

from scripts.lga.cr3bp import CR3BP

def EM2SE(cr3bp, r, theta_s0):
	""" Convertion of the states between the Earth-Moon reference frame 
		and the Sun-Earth reference frame """

	# 1 - Rotation of <pi - theta_s0>°
	R = np.array([[np.cos(np.pi - theta_s0), -np.sin(np.pi - theta_s0), 0,                        0,                         0, 0], 
				  [np.sin(np.pi - theta_s0),  np.cos(np.pi - theta_s0), 0,                        0,                         0, 0],
				  [                       0,                         0, 1,                        0,                         0, 0],
				  [                       0,                         0, 0, np.cos(np.pi - theta_s0), -np.sin(np.pi - theta_s0), 0],
				  [                       0,                         0, 0, np.sin(np.pi - theta_s0),  np.cos(np.pi - theta_s0), 0],
				  [                       0,                         0, 0,                        0,                         0, 1]])

	r = R.dot(r)

	# 2 - Put the center on the Sun
	r += np.array([1-cr3bp.mu, 0, 0, 0, 0, 0])

	return r

def SE2EM(cr3bp, r, theta_s0):
	""" Convertion of the states between the Sun_earth reference frame 
		and the Earth-Moon reference frame """

	# 1 - Put the center on the Earth
	r += np.array([-(1-cr3bp.mu), 0, 0, 0, 0, 0])

	# 2 - Rotation of - <theta_s0>°
	R = np.array([[np.cos(theta_s0 - np.pi), -np.sin(theta_s0 - np.pi), 0,                        0,                         0, 0], 
				  [np.sin(theta_s0 - np.pi),  np.cos(theta_s0 - np.pi), 0,                        0,                         0, 0],
				  [                       0,                         0, 1,                        0,                         0, 0],
				  [                       0,                         0, 0, np.cos(theta_s0 - np.pi), -np.sin(theta_s0 - np.pi), 0],
				  [                       0,                         0, 0, np.sin(theta_s0 - np.pi),  np.cos(theta_s0 - np.pi), 0],
				  [                       0,                         0, 0,                        0,                         0, 1]])
 
	r = R.dot(r)

	return r