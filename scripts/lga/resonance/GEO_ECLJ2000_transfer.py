import numpy as np
import pykep as pk 

import matplotlib.pyplot as plt

from scripts.lga.resonance.apogee_raising import apogee_raising
from scripts.lga.resonance.utils_2 import cart2sph, sph2cart
from scripts.utils import load_bodies, load_kernels

# Spacecraft maximum thrust [N]
T = 1

# Thrust arc semi-angle [°]
eps = 2

# Earth orbit perigee [km]
r_p = 200

# Excess velocity at Moon encounter [km/s]
v_inf = 0.960514


r_ar, t_ar = apogee_raising(T=T/1000, eps=eps*np.pi/180, r_p=r_p, v_inf=v_inf)



# 1 - Loading the SPICE kernels and the planet objects
# ----------------------------------------------------
load_kernels.load()

# Moon departure date (mj2000)
tau = 15128.755128883051

t0 = pk.epoch(tau, julian_date_type='mjd2000')

earth = load_bodies.planet(name='EARTH')
moon  = load_bodies.planet(name='MOON')


# /!\ On est dans le plan orbital de la Lune autour de la Terre, pas dans le HRV /!\


