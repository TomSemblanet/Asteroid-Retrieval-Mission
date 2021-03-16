#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 2021 10:55:00

@author: SEMBLANET Tom

"""

# This file contains the main constants used during the design of a NEA mission.
# The unit of each constant is specified and conversion factors are also provided.

import numpy as np
import math as mt

AU = 1.495978707e11 # Astronomical Unit [m]
G0 = 9.80665 # Standard gravity acceleration at sea level [m/s^2]

# Celestial bodies radius
R_SUN = 696342e3 # Sun radius [m]
R_MERCURY = 2439.4e3 # Mercury radius [m]
R_VENUS = 6051.8e3 # Venus radius [m]
R_EARTH = 6371.0084e3 # Earth radius [m]
R_MOON = 1737.4e3 # Moon radius [m]
R_MARS = 3389.50e3 # Mars radius [m]

# Celestial bodies SMA (semi-major axis)
SMA_MERCURY = 578944e10 # Mercury sma [m]
SMA_VENUS = 1.08159e11 # Venus sma [m]
SMA_EARTH = 1.496e11 # Earth sma [m]
SMA_MOON = 3.84399e8 # Moon sma [m]
SMA_MARS = 2.27987e11 # Mars sma [m]

# Celestial bodies masses
M_SUN = 1.9885e30 # Sun mass [kg]
M_MERCURY = 0.330114e24 # Mercury mass [kg]
M_VENUS = 4.86747e24 # Venus mass [kg]
M_EARTH = 5.97237e24 # Earth mass [kg]
M_MOON = 0.07342e24 # Moon mass [kg]
M_MARS = 0.641712e24 # Mars mass [kg]

# Celestial bodies gravitational parameters (g.p)
MU_SUN = 1.32712440018e20 # Sun g.p [m^3/s^2]
MU_MERCURY = 2.2032e13 # Mercury g.p [m^3/s^2]
MU_VENUS = 3.24859e14 # Venus g.p [m^3/s^2]
MU_EARTH = 3.986004418e14 # Earth  g.p [m^3/s^2]
MU_MOON = 4.9048695e12 # Moon g.p [m^3/s^2]
MU_MARS = 4.282837e13 # Mars g.p [m^3/s^2]

# Average velocities of planets around the Sun
VEL_MERCURY = mt.sqrt(MU_SUN/SMA_MERCURY)
VEL_VENUS = mt.sqrt(MU_SUN/SMA_VENUS)
VEL_EARTH = mt.sqrt(MU_SUN/SMA_EARTH)
VEL_MOON = mt.sqrt(MU_EARTH/SMA_MOON)
VEL_MARS = mt.sqrt(MU_SUN/SMA_MARS)

# Conversion factors
AU2M = 1.496e11
M2AU = 1./AU2M

KM2M = 1e3
M2KM = 1./KM2M

RAD2DEG = 180/mt.pi
DEG2RAD = 1./RAD2DEG

DAY2SEC = 86400
SEC2DAY = 1./DAY2SEC

YEAR2DAY = 365
DAY2YEAR = 1./YEAR2DAY
