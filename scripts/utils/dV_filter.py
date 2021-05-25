import os 
import sys
import numpy as np 
import pykep as pk 
import pygmo as pg 
import pickle

folder_path = sys.argv[1]

files_list = os.listdir(folder_path)

for file in files_list:

	file_path = '/'.join([folder_path, file])
	with open(file_path, 'rb') as file:
		results = pickle.load(file)

	udp = results['udp']
	decision_vector = results['population'].get_x()[0]

	if udp.get_deltaV(x) > 300:
		os.remove(file_path)
		print("File {} removed".format(file_path))
