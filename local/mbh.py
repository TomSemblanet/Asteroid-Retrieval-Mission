#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 2021 09:02:20

@author: SEMBLANET Tom

"""

import os
import sys
import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl
from datetime import datetime as dt

import matplotlib.pyplot as plt

from scripts.utils import load_sqp, load_kernels, load_bodies

from scripts.udp.NEA2Earth_FD_UDP import NEA2Earth

from scripts.utils.post_process import post_process

from data import constants as cst

# Loading the main kernels
load_kernels.load()

# Loading of the initial guess
with open('NEA_Earth_2044_01_04_2021_Earth_Gravity', 'rb') as f:
		init_res = pkl.load(f)

# Get the User-Defined Problem and the initial decision vector
udp = init_res['udp']
x_i = init_res['population'].champion_x

post_process(udp, x_i)

# Define the problem and the algorithm
problem = pg.problem(udp)
inner_algo = load_sqp.load('ipopt')
inner_algo.set_verbosity(0)

# Create the population
population = pg.population(problem, size=0)
population.push_back(x_i)

# Create the MBH
mbh = pg.algorithm(pg.mbh(algo=inner_algo, stop=7))
mbh.set_verbosity(1)

# Evolve
population = mbh.evolve(population)

# 10 - Check feasibility
fitness = udp.fitness(population.champion_x)

# 10 - Check constraints violation
udp.check_con_violation(population.champion_x)

# # 11 - Post process the results
post_process(udp, population.champion_x)