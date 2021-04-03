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

from scripts.udp.NEA2Earth_UDP import NEA2Earth

from scripts.utils.post_process import post_process

from data import constants as cst

# SQP
sqp = sys.argv[1]

# Loading the main kernels
load_kernels.load()

files = os.listdir('/scratch/students/t.semblanet/NEA_Earth_results/' + str(sqp))

for fl in files:
	with open('/scratch/students/t.semblanet/NEA_Earth_results/' + str(sqp) + '/' + str(fl), 'rb') as f:
		res = pkl.load(f)

	population = res['population']
	udp = res['udp']

	print("{} \n----\n\n".format(str(fl)))
	udp.brief(population.champion_x)

