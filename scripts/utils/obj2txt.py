import sys

import pickle as pkl
import pykep as pk
import pygmo as pg
import numpy as np 

if __name__ == '__main__':

	# Path of the Pickle file
	pkl_file_path = sys.argv[1]

	with open(pkl_file_path, 'rb') as file:
		results = pkl.load(file)

	# Extraction of the decision vector
	x = results['population'].get_x()[0]

	# Writing of the decision vector into a file text
	txt_file_nm = '/home/cesure/t.semblanet/Asteroid-Retrieval-Mission/local/txt_files/' + str(round(x[0], 5)) + '.txt'
	np.savetxt(txt_file_nm, x, fmt='%.2f')