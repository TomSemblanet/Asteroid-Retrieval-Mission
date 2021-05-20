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

from src.optimal_control.problem import Problem
from src.optimal_control.optimization import Optimization


class GeocentricTransfer(Problem):
    """ Geocentric Transfer : optimal control problem"""

    def __init__(self):
        """ Initialization of the `GeocentricTransfer` class """
        n_states = 7
        n_controls = 2
        n_path_con = 0
        n_event_con = 9
        n_f_par = 0
        n_nodes = 600

        Problem.__init__(self, n_states, n_controls, n_path_con,
                         n_event_con, n_f_par, n_nodes)

    def set_constants(self):
        """ Setting of the problem constants """
        self.l_c = 42200			   # Characteristic length [km]
        self.t_c = 86400			   # Characteristic time   [s]
        self.m_c = 1000				# Characteristic mass   [kg]

        # earth gravitational parameter [-]   --> 398600.4418 km^3/s^2
        self.mu = 39.59386589622768146796054

        # Initial orbit : circular around the Earth
        # Initial radius				   [-]		 --> r0 * 42200	km
        self.r0 = 1.
        # Final Radius					 [-]		 --> rf * 42200	km
        self.rf = 2.
        # Initial right-ascension		  [-]		 --> 0		°
        self.a0 = 0
        # Initial declinaison			  [-]		 --> 0		°
        self.d0 = 0
        # Initial radial velocity		  [-]		 --> 0		km/s
        self.u0 = 0
        # Initial right-ascencion velocity [-]		 --> Circular orbit
        self.v0 = np.sqrt(self.mu/self.r0)
        # Initial declinaison velocity	 [-]		 --> 0		km/s
        self.w0 = 0

        # Spacecraft
        # Initial mass					  [-]		  --> 1000		kg
        self.m0 = 1
        # Dry mass						  [-]		  --> 10		   kg
        self.m_dry = .001
        # Thrust							[-]		  --> 2e-3		kN
        self.T = .3537895734597
        # Specific impulse				  [-]		  --> 2000		s
        self.isp = 0.02314815
        # Earth sea-level gravity		   [-]		  --> 9.80665e-3  km/s^2
        self.g0 = 1734.745260284
        # Mass flow						 [-]		  --> 1.019716e-4 kg/s
        self.mass_flow = .008810347375

        # Time
        self.tf = 18				  # Final time						  [-]
        self.g_tof = 15				# Guess time of flight

    def set_boundaries(self):
        """ Setting of the states, controls, free-parameters, initial and final times
                        boundaries """
        # x-position [-]
        self.low_bnd.states[0] = 0.9*self.r0
        self.upp_bnd.states[0] = 1.1*self.rf

        # y-position [-]
        self.low_bnd.states[1] = 0
        self.upp_bnd.states[1] = 100 * 3.14

        # z-position [-]
        self.low_bnd.states[2] = 0
        self.upp_bnd.states[2] = 3.14

        # x-velocity [-]
        self.low_bnd.states[3] = 0
        self.upp_bnd.states[3] = 1

        # y-velocity [-]
        self.low_bnd.states[4] = 0
        self.upp_bnd.states[4] = 20

        # z-velocity [-]
        self.low_bnd.states[5] = 0
        self.upp_bnd.states[5] = 10

        # Mass [-]
        self.low_bnd.states[6] = self.m_dry
        self.upp_bnd.states[6] = self.m0

        # Definition of control variables boundaries
        # ------------------------------------------

        # Alpha [rad]
        self.low_bnd.controls[0] = 0
        self.upp_bnd.controls[0] = 2*3.14

        # Beta [rad]
        self.low_bnd.controls[1] = 0
        self.upp_bnd.controls[1] = 3.14

        # Definition of initial / final times
        # -----------------------------------

        # Initial time [-]
        self.low_bnd.ti = self.upp_bnd.ti = 0

        # Final time [-]
        self.low_bnd.tf = 0.1
        self.upp_bnd.tf = self.tf

    def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
        """ Computation of the events constraints """
        events = np.ndarray((self.prm['n_event_con'], 1),
                            dtype=cppad_py.a_double)

        # Initial conditions
        ri = xi[0]  # Radius [-]
        ai = xi[1]  # Right-ascension [rad]
        di = xi[2]  # Declinaison [rad]
        ui = xi[3]  # Radial-velocity [-]
        vi = xi[4]  # Right-ascension velocity [-]
        wi = xi[5]  # Declinaison velocity [-]
        mi = xi[6]  # Mass [kg]

        # Final conditions
        v_norm_2 = (xf[3]*xf[3] + xf[4]*xf[4] + xf[5] *
                    xf[5])  # squared final velocity norm
        circ = xf[0] * v_norm_2  # final orbit : circular

        r_f = xf[0]  # Final Radius [-]

        events[0] = ri  # Initial radius
        events[1] = ai  # Initial right-ascension
        events[2] = di  # Initial declinaison
        events[3] = ui  # Initial radial-velocity
        events[4] = vi  # Initial right-ascension velocity
        events[5] = wi  # Initial declinaison velocity
        events[6] = mi  # Initial mass

        events[7] = circ  # Final circularity
        events[8] = r_f  # Final radius

        return events

    def set_events_constraints_boundaries(self):
        """ Setting of the events constraints boundaries """
        # Initial radius [-]
        self.low_bnd.event[0] = self.upp_bnd.event[0] = self.r0

        # Initial right-ascension [rad]
        self.low_bnd.event[1] = self.upp_bnd.event[1] = self.a0

        # Initial declinaison [rad]
        self.low_bnd.event[2] = self.upp_bnd.event[2] = self.d0

        # Initial radial-velocity [-]
        self.low_bnd.event[3] = self.upp_bnd.event[3] = self.u0

        # Initial right-ascension velocity [-]
        self.low_bnd.event[4] = self.upp_bnd.event[4] = self.v0

        # Initial declinaison velocity [-]
        self.low_bnd.event[5] = self.upp_bnd.event[5] = self.w0

        # Initial mass [kg]
        self.low_bnd.event[6] = self.upp_bnd.event[6] = self.m0

        # Final circularity [-]
        self.low_bnd.event[7] = self.upp_bnd.event[7] = self.mu

        # Final radius [-]
        self.low_bnd.event[8] = self.upp_bnd.event[8] = self.rf

    def dynamics(self, states, controls, f_prm, expl_int=False):
        """ Computation of the states derivatives """
        if expl_int == False:
            dynamics = np.ndarray(
                (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
        else:
            dynamics = np.zeros(len(states))

        # States
        # ======

        # Radius [-]
        r = states[0]

        # Right-ascension [rad]
        a = states[1]

        # Declinaison [rad]
        d = states[2]

        # Radial velocity [-]
        u = states[3]

        # Right-ascension velocity [-]
        v = states[4]

        # Declinaison velocity [-]
        w = states[5]

        # Mass [kg]
        m = states[6]

        # Controls
        # ========

        # Alpha [rad]
        alpha = controls[0]

        # Beta [rad]
        beta = controls[1]

        # Derivatives
        # ===========

        # Radius derivative [-]
        r_dot = u

        # Right-ascension derivative [rad]
        a_dot = v/r

        # Declinaison derivative [rad]
        d_dot = w/r

        # Avoid multiple function call
        cos_d = np.cos(d)
        cos_a = np.cos(alpha)
        cos_b = np.cos(beta)

        sin_d = np.sin(d)
        sin_a = np.sin(alpha)
        sin_b = np.sin(beta)

        # Radial velocity derivative [-]
        u_dot = v*v * cos_d*cos_d / r + w*w / \
            r - self.mu/(r*r) + self.T*cos_a*cos_b/m

        # Right-ascension velocity derivative [-]
        v_dot = -2*u*v/r + 2*v*w * \
            (sin_d/cos_d)/r + self.T*sin_a * \
            cos_b/(m*cos_d)

        # Declinaison velocity derivative [-]
        w_dot = -2*u*w/r - w*w * \
            sin_d*cos_d/r + self.T*sin_b/m

        # Mass derivative [kg]
        m_dot = - self.mass_flow

        dynamics[0] = r_dot  # Radius derivative
        dynamics[1] = a_dot  # Right-ascension derivative
        dynamics[2] = d_dot  # Declinaison derivative
        dynamics[3] = u_dot  # Radial velocity derivative
        dynamics[4] = v_dot  # Right-ascension velocity derivative
        dynamics[5] = w_dot  # Declinaison velocity derivative
        dynamics[6] = m_dot  # Mass derivative

        return dynamics

    def end_point_cost(self, ti, xi, tf, xf, f_prm):
        """ Computation of the end point cost (Mayer term) """
        mf = xf[6]
        return - mf

    def set_initial_guess(self):
        """ Setting of the initial guess for the states, controls, free-parameters
                        and time grid """

        # Time
        # ====
        self.initial_guess.time = np.linspace(
            0, self.g_tof, self.prm['n_nodes'])

        # States
        # ======
        self.initial_guess.states = np.ndarray(
            shape=(self.prm['n_states'], self.prm['n_nodes']))

        # Radius [-]
        self.initial_guess.states[0] = np.linspace(
            self.r0, self.rf, self.prm['n_nodes'])

        # Right-ascension [rad]
        self.initial_guess.states[1] = np.linspace(
            0, self.v0/self.r0 * self.g_tof, self.prm['n_nodes'])

        # Declinaison [rad]
        self.initial_guess.states[2] = np.linspace(
            0, 0, self.prm['n_nodes'])

        # Radial velocity [-]
        self.initial_guess.states[3] = np.linspace(
            self.u0, self.u0, self.prm['n_nodes'])

        # Right-ascension velocity [-]
        self.initial_guess.states[4] = np.linspace(
            self.v0, self.v0, self.prm['n_nodes'])

        # Declinaison velocity [-]
        self.initial_guess.states[5] = np.linspace(
            self.w0, self.w0, self.prm['n_nodes'])

        # Mass [kg]
        self.initial_guess.states[6] = np.linspace(
            self.m0, max(self.m0 - self.g_tof * self.mass_flow, self.m_dry), self.prm['n_nodes'])

        # Controls
        # ========
        self.initial_guess.controls = np.ndarray(
            shape=(self.prm['n_controls'], self.prm['n_nodes']))

        # Alpha [rad]
        self.initial_guess.controls[0] = np.linspace(
            1.5, 1.5, self.prm['n_nodes'])

        # Beta [rad]
        self.initial_guess.controls[1] = np.linspace(0, 0, self.prm['n_nodes'])


if __name__ == '__main__':

    # Instantiation of the problem
    problem = GeocentricTransfer()

    # Instantiation of the optimization
    optimization = Optimization(problem=problem)

    # Launch of the optimization
    optimization.run()
