import os 
import sys
import pickle
import numpy as np 
import pykep as pk 
import pygmo as pg 

from scripts.utils import load_sqp, load_kernels, load_bodies

load_kernels.load()

folder_path = sys.argv[1]

files_list = os.listdir(folder_path)

for file in files_list:

	file_path = '/'.join([folder_path, file])
	with open(file_path, 'rb') as file:
		results = pickle.load(file)

	udp = results['udp']
	decision_vector = results['population'].get_x()[0]

	if udp.get_deltaV(decision_vector) > 300:
		os.remove(file_path)
		print("File {} removed".format(file_path))
