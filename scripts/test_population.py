import os
import pykep as pk 
import pygmo as pg 
import pickle as pkl

from scripts.loaders import load_sqp, load_kernels, load_bodies

from scripts.post_process import post_process

with open('res', 'rb') as f:
	res = pkl.load(f)

load_kernels.load()

udp = res['udp']
population = res['population']

print(population.)

# post_process(udp, population.champion_x)