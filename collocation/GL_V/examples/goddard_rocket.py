#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  14 09:50:23 2020

@author: SEMBLANET Tom

"""

import os
import cppad_py
import math
import numpy as np
import matplotlib.pyplot as plt

from collocation.GL_V.src.problem import Problem
from collocation.GL_V.src.optimization import Optimization


class GoddardRocket(Problem):
	""" Goddard Rocket optimal control problem"""

	def __init__(self):
		""" Initialization of the `GoddardRocket` class """
		n_states = 3
		n_controls = 1
		n_st_path_con = 0
		n_ct_path_con = 0
		n_event_con = 4
		n_f_par = 0
		n_nodes = 300

		Problem.__init__(self, n_states, n_controls, n_st_path_con, n_ct_path_con, 
						 n_event_con, n_f_par, n_nodes)

	def set_constants(self):
		""" Setting of the problem constants """
		self.D0 = 310
		self.beta = 500
		self.c = 0.5

	def set_boundaries(self):
		""" Setting of the states, controls, free-parameters, initial and final times
						boundaries """
		# States boundaries
		self.low_bnd.states[0] = 1
		self.upp_bnd.states[0] = 2

		self.low_bnd.states[1] = 0
		self.upp_bnd.states[1] = 0.5

		self.low_bnd.states[2] = 0.6
		self.upp_bnd.states[2] = 1

		# Controls boundaries
		self.low_bnd.controls[0] = 0
		self.upp_bnd.controls[0] = 3.5

		# Initial and final times boundaries
		self.low_bnd.ti = self.upp_bnd.ti = 0
		self.low_bnd.tf = 0.1
		self.upp_bnd.tf = 1

	def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
		""" Computation of the events constraints """
		events = np.ndarray((self.prm['n_event_con'], 1),
							dtype=cppad_py.a_double)

		h_i = xi[0]
		v_i = xi[1]
		m_i = xi[2]

		m_f = xf[2]

		events[0] = h_i
		events[1] = v_i
		events[2] = m_i
		events[3] = m_f

		return events

	def set_events_constraints_boundaries(self):
		""" Setting of the events constraints boundaries """
		self.low_bnd.event[0] = self.upp_bnd.event[0] = 1

		self.low_bnd.event[1] = self.upp_bnd.event[1] = 0

		self.low_bnd.event[2] = self.upp_bnd.event[2] = 1

		self.low_bnd.event[3] = self.upp_bnd.event[3] = 0.6

	def dynamics(self, states, controls, f_prm, expl_int=False):
		""" Computation of the states derivatives """
		if expl_int == False:
			dynamics = np.ndarray(
				(states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
		else:
			dynamics = np.zeros(len(states))

		h = states[0]
		v = states[1]
		m = states[2]

		T = controls[0]

		D = self.D0 * v * v * np.exp(-self.beta*h)

		g = 1 / (h*h)

		h_dot = v
		v_dot = (T - D)/m - g
		m_dot = -T/self.c

		dynamics[0] = h_dot
		dynamics[1] = v_dot
		dynamics[2] = m_dot

		return dynamics

	def end_point_cost(self, ti, xi, tf, xf, f_prm):
		""" Computation of the end point cost (Mayer term) """
		hf = xf[0]
		return - hf

	def set_initial_guess(self):
		""" Setting of the initial guess for the states, controls, free-parameters
						and time grid """

		# Time
		self.initial_guess.time = np.linspace(0, 15, self.prm['n_nodes'])

		# States
		self.initial_guess.states = np.ndarray(
			shape=(self.prm['n_states'], self.prm['n_nodes']))

		self.initial_guess.states[0] = np.array(
			[1 for _ in range(self.prm['n_nodes'])])
		self.initial_guess.states[1] = np.array(
			[0 for _ in range(self.prm['n_nodes'])])
		self.initial_guess.states[2] = np.array(
			[1 for _ in range(self.prm['n_nodes'])])

		# Controls
		self.initial_guess.controls = np.ndarray(
			shape=(self.prm['n_controls'], self.prm['n_nodes']))

		self.initial_guess.controls[0] = np.array(
			[3.5 for _ in range(self.prm['n_nodes'])])


if __name__ == '__main__':

	# Instantiation of the problem
	problem = GoddardRocket()

	# Instantiation of the optimization
	optimization = Optimization(problem=problem)

	# Launch of the optimization
	optimization.run()
