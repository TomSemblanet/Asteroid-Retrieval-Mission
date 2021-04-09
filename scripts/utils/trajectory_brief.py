#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 08 2021 21:02:20

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
from scripts.udp.NEA_Earth_UDP import NEA2Earth
from scripts.utils.post_process import post_process
from data import constants as cst

# Load the main kernels
load_kernels.load()

# Path to the Pickle file
file_path = str(sys.argv[1])

# Open the file and extract its content
with open(file_path, 'rb') as file:
	results = pkl.load(file)

# Call the `brief` method of the UDP
results['udp'].brief(results['population'].get_x()[0])






