import pykep as pk 
import pygmo as pg 
import pickle as pkl
from datetime import datetime as dt

from scripts.loaders import load_sqp, load_kernels, load_bodies

from scripts.UDP.Earth2NEA_UDP import Earth2NEA

from data import constants as cst


# Loading the main kernels
load_kernels.load()

# Loading of the target asteroid
ast = load_bodies.asteroid('2020 CD3')

# 2 - Launch window
lw_low = pk.epoch_from_string('2040-01-01 00:00:00')
lw_upp = pk.epoch_from_string('2044-12-31 23:59:59')

# 3 - Time of flight
tof_low = cst.YEAR2DAY * 0.01
tof_upp = cst.YEAR2DAY * 3.00

# 4 - Spacecraft
m0 = 2000
Tmax = 1
Isp = 2500

# 5 - Velocity at infinity
vinf_max = 2.5e3

# 5 - Optimization algorithm
algorithm = load_sqp.load('slsqp')
algorithm.extract(pg.nlopt).maxeval = 20

# 6 - Problem
udp = Earth2NEA(nea=ast, n_seg=30, t0=(lw_low, lw_upp), \
	tof=(tof_low, tof_upp), m0=m0, Tmax=Tmax, Isp=Isp, vinf_max=vinf_max)

problem = pg.problem(udp)
problem.c_tol = [1e-8] * problem.get_nc()


# 7 - Creation of an island
isl = pg.island(algo=algorithm, prob=udp, size=2, udi=pg.ipyparallel_island(), seed=1230)
isl.evolve()

isl.wait_check()

# # 7 - Population
# population = pg.population(problem, size=1, seed=123)

# # 8 - Optimization
# population = algorithm.evolve(population)