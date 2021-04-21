import pickle as pkl
import pykep as pk
import pygmo as pg 

if __name__ == '__main__':

	# Path of the file containing the decision vector
	txt_file_path = sys.argv[1]

	if 'NEA_Earth' in txt_file_path:
		
		
		