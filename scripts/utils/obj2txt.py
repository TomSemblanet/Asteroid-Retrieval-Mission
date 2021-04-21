import pickle as pkl
import pykep as pk
import pygmo as pg 

if __name__ == '__main__':

	# Path of the Pickle file
	pkl_file_path = sys.argv[1]

	with open(pkl_file_path, 'rb') as file:
		results = pkl.load(file)

	# Extraction of the decision vector
	x = results['population'].get_x()[0]

	print(x)