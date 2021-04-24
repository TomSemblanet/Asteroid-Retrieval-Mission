#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 2021 09:02:20

@author: SEMBLANET Tom

"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# File to open 
file_path = sys.argv[1]

# Opening and reading the file
file = open(file_path, 'r')
lines = file.readlines()

# Array containing the average delta-v for each years
years = np.arange(2025, 2049)
dVs = np.zeros(len(lines))

for line in lines:
	year = int(line[5:9])
	index = year - 2025

	dVs[index] = float(line[12:16])

plt.bar(years, dVs / 1000)

plt.xlabel("Launch year")
plt.ylabel("Score")

plt.show()
