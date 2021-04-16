import pickle as pkl
import pykep as pk
import pygmo as pg 

if __name__ == '__main__':

	# Path of the file containing the decision vector
	txt_file_path = sys.argv[1]

	# Type of mission (either NEA_Earth or Earth_NEA)
	mission = sys.argv[2]

	if mission == 'NEA_Earth':
		