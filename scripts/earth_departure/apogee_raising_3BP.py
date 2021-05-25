import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.earth_departure.utils import kepler_thrust, kep2cart, cart2kep, moon_reached, R2, apside_pass, plot_env_2D, thrust_ignition, angle_w_Ox
from scripts.earth_departure import constants as cst

