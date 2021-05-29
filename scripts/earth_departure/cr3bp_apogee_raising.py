import sys
import numpy as np
import pykep as pk 
import pickle

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from scripts.earth_departure.apogee_raising import apogee_raising
from scripts.earth_departure.utils import cart2sph, sph2cart, cr3bp_moon_approach, kep2cart, R2
from scripts.earth_departure.cr3bp import CR3BP
from scripts.earth_departure import constants as cst
from scripts.utils import load_bodies, load_kernels

from scripts.earth_departure.OCP_moon_moon_leg import MoonMoonLeg
from scripts.earth_departure.OCP_apogee_raising import ApogeeRaising
from collocation.GL_V.src.optimization import Optimization


def cr3bp_dynamics_thrusted(t, r, cr3bp, Tmax, mass):

	x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot = cr3bp.states_derivatives(t, r)

	# S/C velocity magnitude 
	v_mag = np.linalg.norm(r[3:])

	vx_dot += Tmax / mass * r[3] / v_mag
	vy_dot += Tmax / mass * r[4] / v_mag
	vz_dot += Tmax / mass * r[5] / v_mag

	return np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])

def cr3bp_apogee_raising(Tmax, mass, r_p, r_a, v_inf):

	mu = 0.012151
	L = 384400
	T = 2360591.424
	V = L/(T/(2*np.pi))
	cr3bp = CR3BP(mu=mu, L=L, V=V, T=T/(2*np.pi))

	Tmax *= (T/(2*np.pi))**2 / L


	# 1 - Definition of the initial circular orbit and S/C initial state in the Earth inertial frame
	# ----------------------------------------------------------------------------------------------
	a  = (2*cst.R_E + r_p + r_a) / 2		 # SMA [km]
	e  = 1 - (cst.R_E + r_p) / a			 # Eccentricity [-]
	i  = 0							 	 	 # Inclinaison [rad]
	W  = 0				 				 	 # RAAN [rad]
	w  = np.pi			 				 	 # Perigee anomaly [rad]
	ta = 0					  				 # True anomaly [rad]


	# S/C initial states on its circular orbit around the Earth [km] | [km/s]
	r0_eci = kep2cart(a, e, i, W, w, ta, cst.mu_E)
	r0_eci[:3] /= L
	r0_eci[3:] /= V

	theta = 0 * np.pi / 180

	r0_eci[:3] = R2(theta).dot(r0_eci[:3])
	r0_eci[3:] = R2(theta).dot(r0_eci[3:])

	# S/C initial states on its circular orbit around the Earth in the synodic frame
	r0_syn = cr3bp.eci2syn(t=0, r=r0_eci)

	t_span = [0, 10]
	t_eval = np.linspace(t_span[0], t_span[-1], 10000)

	propagation = solve_ivp(fun=cr3bp_dynamics_thrusted, t_span=t_span, t_eval=t_eval, y0=r0_syn, args=(cr3bp, Tmax, mass), rtol=1e-12, atol=1e-12)

	for k, t in enumerate(propagation.t):
		propagation.y[:, k] = cr3bp.syn2eci(t=t, r=propagation.y[:, k])

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(propagation.y[0], propagation.y[1], '-', color='blue', linewidth=1)

	ax.set_xlim(-2, 2)
	ax.set_ylim(-2, 2)

	plt.grid()
	plt.show()

