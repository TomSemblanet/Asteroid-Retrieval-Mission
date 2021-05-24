import sys
import pickle 
import numpy as np 
import matplotlib.pyplot as plt

import cppad_py

from collocation.GL_V.src.problem import Problem
from collocation.GL_V.src.optimization import Optimization

from scripts.earth_departure import constants as cst

class MoonFlyBy(Problem):
	""" CR3BP : Moon-Moon Leg optimal control problem """

	def __init__(self, cr3bp, mass0, Tmax, fwd_trajectory, fwd_time, bwd_trajectory, bwd_time, r_m):
		""" Initialization of the `GoddardRocket` class """
		n_states = 7
		n_controls = 4
		n_st_path_con = 0
		n_ct_path_con = 1
		n_event_con = 13
		n_f_par = 0
		n_nodes = 200

		Problem.__init__(self, n_states, n_controls, n_st_path_con, n_ct_path_con, 
						 n_event_con, n_f_par, n_nodes)

		# Set some attributs
		self.cr3bp = cr3bp 

		self.mass0 = mass0 # [kg]
		self.Tmax = Tmax   # [kN]

		self.fwd_trajectory = fwd_trajectory # [L] | [L/T]
		self.fwd_time = fwd_time # [T]

		self.bwd_trajectory = bwd_trajectory # [L] | [L/T]
		self.bwd_time = bwd_time # [T]

		self.r_m = r_m # [km]

	def set_constants(self):
		""" Setting of the problem constants """
		self.Tmax /= self.cr3bp.L / self.cr3bp.T**2   # Thrusts dimensioning

		self.r_m = (self.r_m + cst.R_M) / self.cr3bp.L

		self.g0 = 9.80665e-3 / (self.cr3bp.L / self.cr3bp.T**2)
		self.Isp = 2000 / self.cr3bp.T

	def set_boundaries(self):
		""" Setting of the states, controls, free-parameters, initial and final times
						boundaries """

		# States boundaries
		# X [-]
		self.low_bnd.states[0] = -2
		self.upp_bnd.states[0] =  2

		# Y [-]
		self.low_bnd.states[1] = -2
		self.upp_bnd.states[1] =  2

		# Z [-]
		self.low_bnd.states[2] = -2
		self.upp_bnd.states[2] =  2

		# Vx [-]
		self.low_bnd.states[3] = -20
		self.upp_bnd.states[3] =  20

		# Vy [-]
		self.low_bnd.states[4] = -20
		self.upp_bnd.states[4] =  20

		# Vz [-]
		self.low_bnd.states[5] = -20
		self.upp_bnd.states[5] =  20

		# m [kg]
		self.low_bnd.states[6] = 1e-6
		self.upp_bnd.states[6] = self.mass0


		# T [-]
		self.low_bnd.controls[0] = 1e-6
		self.upp_bnd.controls[0] = self.Tmax

  		# Tx [-]
		self.low_bnd.controls[1] = - 1
		self.upp_bnd.controls[1] =   1

		# Ty [-]
		self.low_bnd.controls[2] = - 1
		self.upp_bnd.controls[2] =   1

		# Tz [-]
		self.low_bnd.controls[3] = - 1
		self.upp_bnd.controls[3] =   1


		# Initial and final times boundaries
		self.low_bnd.ti = self.upp_bnd.ti = self.fwd_time[0]
		self.low_bnd.tf = 0.2 * self.bwd_time[-1]
		self.upp_bnd.tf = 2.0 * self.bwd_time[-1]


	def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
		""" Computation of the events constraints """
		events = np.ndarray((self.prm['n_event_con'], 1),
							dtype=cppad_py.a_double)

		x_i, y_i, z_i, vx_i, vy_i, vz_i, m_i = xi
		x_f, y_f, z_f, vx_f, vy_f, vz_f, _   = xf

		events[0] = x_i  - self.fwd_trajectory[0, 0]
		events[1] = y_i  - self.fwd_trajectory[1, 0]
		events[2] = z_i  - self.fwd_trajectory[2, 0]
		events[3] = vx_i - self.fwd_trajectory[3, 0]
		events[4] = vy_i - self.fwd_trajectory[4, 0]
		events[5] = vz_i - self.fwd_trajectory[5, 0]

		events[6]  = x_f  - self.bwd_trajectory[0, -1]
		events[7]  = y_f  - self.bwd_trajectory[1, -1]
		events[8]  = z_f  - self.bwd_trajectory[2, -1]
		events[9]  = vx_f - self.bwd_trajectory[3, -1]
		events[10] = vy_f - self.bwd_trajectory[4, -1]
		events[11] = vz_f - self.bwd_trajectory[5, -1]

		events[12] = m_i - self.mass0

		return events


	def set_events_constraints_boundaries(self):
		""" Setting of the events constraints boundaries """
		self.low_bnd.event[0] = self.upp_bnd.event[0] = 0
		self.low_bnd.event[1] = self.upp_bnd.event[1] = 0
		self.low_bnd.event[2] = self.upp_bnd.event[2] = 0
		self.low_bnd.event[3] = self.upp_bnd.event[3] = 0
		self.low_bnd.event[4] = self.upp_bnd.event[4] = 0
		self.low_bnd.event[5] = self.upp_bnd.event[5] = 0

		self.low_bnd.event[6] = self.upp_bnd.event[6] = 0
		self.low_bnd.event[7] = self.upp_bnd.event[7] = 0
		self.low_bnd.event[8] = self.upp_bnd.event[8] = 0
		self.low_bnd.event[9] = self.upp_bnd.event[9] = 0
		self.low_bnd.event[10] = self.upp_bnd.event[10] = 0
		self.low_bnd.event[11] = self.upp_bnd.event[11] = 0

		self.low_bnd.event[12] = self.upp_bnd.event[12] = 0

	def path_constraints(self, states, controls, states_add, controls_add, controls_col, f_par):

		st_path = np.ndarray((self.prm['n_st_path_con'],
							2*self.prm['n_nodes']-1), dtype=cppad_py.a_double)
		ct_path = np.ndarray((self.prm['n_ct_path_con'],
							4*self.prm['n_nodes']-3), dtype=cppad_py.a_double)

		# Thrust magnitude in x, y and z directions in the synodic frame [-]
		ux = np.concatenate((controls[1], controls_add[1], controls_col[1]))
		uy = np.concatenate((controls[2], controls_add[2], controls_col[2]))
		uz = np.concatenate((controls[3], controls_add[3], controls_col[3]))

		# # S/C position in the synodic frame [-]
		# x = np.concatenate((states[1], states_add[1]))
		# y = np.concatenate((states[2], states_add[2]))
		# z = np.concatenate((states[3], states_add[3]))

		# # S/C - Moon distance
		# d2 = (x - (1-self.cr3bp.mu))*(x - (1-self.cr3bp.mu)) + y*y + z*z
		# st_path[0] = d2

		u2 = ux*ux + uy*uy + uz*uz

		ct_path[0] = u2 - 1

		return st_path, ct_path

	def set_path_constraints_boundaries(self):
		""" Setting of the path constraints boundaries """
		# self.low_bnd.st_path[0] = self.r_m**2
		# self.upp_bnd.st_path[0] = 2

		self.low_bnd.ct_path[0] = self.upp_bnd.ct_path[0] = 0



	def dynamics(self, states, controls, f_prm, expl_int=False):
		""" Computation of the states derivatives """
		if expl_int == False:
			dynamics = np.ndarray(
				(states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
		else:
			dynamics = np.zeros(len(states))

		# Mass [kg]
		m = states[6]

		# Extraction of controls
		T = controls[0]
		ux, uy, uz = controls[1:]

		x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot = self.cr3bp.states_derivatives(0, states[:-1])

		dynamics[0] = x_dot
		dynamics[1] = y_dot
		dynamics[2] = z_dot

		dynamics[3] = vx_dot + T / m * ux
		dynamics[4] = vy_dot + T / m * uy
		dynamics[5] = vz_dot + T / m * uz

		dynamics[6] = - T / self.Isp / self.g0

		return dynamics

	def end_point_cost(self, ti, xi, tf, xf, f_prm):
		""" Computation of the end point cost (Mayer term) """
		mf = xf[-1]
		return - mf / self.mass0


	def set_initial_guess(self):
		""" Setting of the initial guess for the states, controls, free-parameters
						and time grid """

		# Sampling of the states and time
		time_smpld = np.zeros((6, self.prm['n_nodes']))
		trajectory_smpld = np.zeros((1, self.prm['n_nodes']))

		step = int((len(self.fwd_time) + len(self.bwd_time)) / self.prm['n_nodes']) + 3

		fwd_time_smpld = self.fwd_time[0::step]
		bwd_time_smpld = self.bwd_time[0::step]

		fwd_trajectory_smpld = self.fwd_trajectory[:, 0::step]
		bwd_trajectory_smpld = self.bwd_trajectory[:, 0::step]

		n_add_nodes = self.prm['n_nodes'] - np.hstack((fwd_time_smpld, bwd_time_smpld)).shape[0]
		add_time = np.linspace(fwd_time_smpld[-1], bwd_time_smpld[0], n_add_nodes)
		time_smpld = np.hstack((fwd_time_smpld, add_time, bwd_time_smpld))

		if n_add_nodes % 2 == 1:
			n_fwd = int(n_add_nodes / 2) + 1
			n_bwd = int(n_add_nodes / 2) 

		else:
			n_fwd = n_bwd = int(n_add_nodes / 2)

		trajectory_smpld = np.hstack((fwd_trajectory_smpld, np.tile(np.reshape(fwd_trajectory_smpld[:, -1], (6, 1)), n_fwd), \
			np.tile(np.reshape(bwd_trajectory_smpld[:,  0], (6, 1)), n_bwd), bwd_trajectory_smpld))
		

		# Time
		self.initial_guess.time = np.linspace(time_smpld[0], time_smpld[-1], self.prm['n_nodes'])

		# States
		self.initial_guess.states = np.ndarray(
			shape=(self.prm['n_states'], self.prm['n_nodes']))

		self.initial_guess.states[0] = trajectory_smpld[0]
		self.initial_guess.states[1] = trajectory_smpld[1]
		self.initial_guess.states[2] = trajectory_smpld[2]

		self.initial_guess.states[3] = trajectory_smpld[3]
		self.initial_guess.states[4] = trajectory_smpld[4]
		self.initial_guess.states[5] = trajectory_smpld[5]

		self.initial_guess.states[6] = self.mass0 * np.ones(self.prm['n_nodes'])

		# Controls
		self.initial_guess.controls = np.ndarray(
			shape=(self.prm['n_controls'], self.prm['n_nodes']))

		self.initial_guess.controls = np.zeros((4, self.prm['n_nodes']))

