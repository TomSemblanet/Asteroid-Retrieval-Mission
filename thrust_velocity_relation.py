#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 2021 09:02:20

@author: SEMBLANET Tom

"""

import os
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

# Loading the main kernels
load_kernels.load()

with open('NEA_Earth_2042', 'rb') as f:
		res = pkl.load(f)

udp = res['udp']
population = res['population']
x = population.champion_x

# Velocity
# --------
rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, _, _, _, _, _, _ = udp.propagate(x)

v = np.concatenate((vfwd, vbwd))

for v_ in v:
	v_ /= np.linalg.norm(v_)

vx = v[:, 0]
vy = v[:, 1]
vz = v[:, 2]

# Thrust
# ------

thrust = np.reshape(x[3:], (30, 3))

ux = thrust[:, 0]
uy = thrust[:, 1]
uz = thrust[:, 2]

plt.plot(vx, label='Vx')
plt.plot(vy, label='Vy')
plt.plot(vz, label='Vz')

plt.plot(ux, label='Ux')
plt.plot(uy, label='Uy')
plt.plot(uz, label='Uz')


plt.legend()
plt.grid()
plt.show()















