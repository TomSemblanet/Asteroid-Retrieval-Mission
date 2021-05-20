#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  14 09:50:23 2020

@author: SEMBLANET Tom

"""
 
import cppad_py
import numpy

from src.problem import Problem
from src.optimization import Optimization


class PlanarHeliocentricTransfer(Problem):
    """ Planar Heliocentric Transfer optimal control problem """

    def __init__(self):
        """ Initialization of the `VanDerPolOscillator` class """
        n_states = 5
        n_controls = 1
        n_st_path_con = 0
        n_ct_path_con = 0
        n_event_con = 7
        n_f_par = 0
        n_nodes = 400

        Problem.__init__(self, n_states, n_controls, n_st_path_con, n_ct_path_con,
                         n_event_con, n_f_par, n_nodes)

    def set_constants(self):
        """ Setting of the problem constants """

        # sun gravitational parameter [UA^3/day^2]
        self.mu = 0.000295912208232212800652365

        self.r0 = 1                  # [UA]                       Initial radius
        self.rf = 2                  # [UA]                       Final radius
        self.l0 = 0                  # [rad]                      Initial longitude
        self.u0 = 0                  # [UA/day]                   Initial radial velocity
        self.v0 = 0.017202098948448  # [UA/day]                   Initial tangential velocity
        self.m0 = 4535.9237          # [kg]                       Initial mass

        self.T = 0.385554            # [kg.UA/day^2]              Thrust
        self.mass_flow = 6.304934    # [kg/day]                   Mass flow    --> 0.263 kg/h

        self.tf = 200                # [days]                     Final time
        self.g_tof = 140             # [days]                    Guessed time of flight


    def set_boundaries(self):
        """ Setting of the states, controls, free-parameters, initial and final times
                        boundaries """
        # r [UA]
        self.low_bnd.states[0] = 0.9*self.r0
        self.upp_bnd.states[0] = 1.1*self.rf

        # l [rad]
        self.low_bnd.states[1] = 0
        self.upp_bnd.states[1] = 4*numpy.pi

        # u [UA/day]
        self.low_bnd.states[2] = 0
        self.upp_bnd.states[2] = 0.08

        # v [UA/day]
        self.low_bnd.states[3] = 0
        self.upp_bnd.states[3] = 0.08

        # m [kg]
        self.low_bnd.states[4] = 3000
        self.upp_bnd.states[4] = self.m0

        # Definition of control variables boundaries
        # ------------------------------------------

        # Direction [rad]
        self.low_bnd.controls[0] = 0
        self.upp_bnd.controls[0] = 2*numpy.pi

        # Definition of initial / final times
        # -----------------------------------

        # Initial time [day]
        self.low_bnd.ti = self.upp_bnd.ti = 0

        # Final time [day]
        self.low_bnd.tf = 0.1
        self.upp_bnd.tf = self.tf

    def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
        """ Computation of the events constraints """
        events = numpy.ndarray((self.prm['n_event_con'], 1),
                               dtype=cppad_py.a_double)

        # Initial conditions
        ri = xi[0]              # radius [UA]
        li = xi[1]              # longitude [rad]
        ui = xi[2]              # radial velocity [UA/day]
        vi = xi[3]              # tangential velocity [UA/day]
        mi = xi[4]              # mass [kg]

        v_norm_2 = (xf[2]*xf[2] + xf[3]*xf[3])  # squared final velocity norm

        circ = xf[0] * v_norm_2  # final orbit : circular

        r_f = xf[0]

        events[0] = ri  # Initial radius
        events[1] = li  # Initial longitude
        events[2] = ui  # Initial radial velocity
        events[3] = vi  # Initial tangential velocity
        events[4] = mi  # Initial mass

        events[5] = circ  # Final circularity
        events[6] = r_f   # Final radius

        return events

    def set_events_constraints_boundaries(self):
        """ Setting of the events constraints boundaries """
        # Initial radius [UA]
        self.low_bnd.event[0] = self.upp_bnd.event[0] = self.r0

        # Initial longitude [rad]
        self.low_bnd.event[1] = self.upp_bnd.event[1] = self.l0

        # Initial radial velocity [UA/day]
        self.low_bnd.event[2] = self.upp_bnd.event[2] = self.u0

        # Initial tangential velocity [UA/day]
        self.low_bnd.event[3] = self.upp_bnd.event[3] = self.v0

        # Initial mass [kg]
        self.low_bnd.event[4] = self.upp_bnd.event[4] = self.m0

        # Final orbit : circular
        self.low_bnd.event[5] = self.upp_bnd.event[5] = self.mu

        # Final radius
        self.low_bnd.event[6] = self.upp_bnd.event[6] = self.rf

    def dynamics(self, states, controls, f_prm, expl_int=False):
        """ Computation of the states derivatives """
        if expl_int == False:
            dynamics = numpy.ndarray(
                (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
        else:
            dynamics = numpy.zeros(len(states))

        # Radius [UA]
        r = states[0]

        # Longitude [rad]
        l = states[1]  

        # Radial velocity [UA/day]
        u = states[2]

        # Tangential velocity [UA/day]
        v = states[3]

        # Mass [kg]
        m = states[4]

        # Direction [rad]
        phi = controls[0]

        # Radius derivative [UA/day]
        r_dot = u

        # Longitude derivative [rad/day]
        l_dot = v/r

        # Radial velocity derivative [UA/day^2]
        u_dot = v*v/r - self.mu/(r*r) + self.T*numpy.sin(phi)/m

        # Tangential velocity derivative [rad/day^2]
        v_dot = u*v/r + self.T*numpy.cos(phi)/m

        # Mass derivative [kg/day]
        m_dot = -self.mass_flow

        dynamics[0] = r_dot
        dynamics[1] = l_dot
        dynamics[2] = u_dot
        dynamics[3] = v_dot
        dynamics[4] = m_dot

        return dynamics

    def end_point_cost(self, ti, xi, tf, xf, f_prm):
        """ Computation of the end point cost (Mayer term) """
        mf = xf[-1]
        # Maximize the final radius
        return - mf / self.m0

    def set_initial_guess(self):
        """ Setting of the initial guess for the states, controls, free-parameters
                        and time grid """
        # Time
        # ====
        self.initial_guess.time = numpy.linspace(0, self.g_tof, self.prm['n_nodes'])

        # States
        # ======
        self.initial_guess.states = numpy.ndarray(
            shape=(self.prm['n_states'], self.prm['n_nodes']))

        # Radius [UA]
        self.initial_guess.states[0] = numpy.linspace(
            self.r0, self.rf, self.prm['n_nodes'])

        # Longitude [rad]
        self.initial_guess.states[1] = numpy.linspace(
            0, self.v0/self.r0 * self.g_tof, self.prm['n_nodes'])

        # Radial velocity [UA/day]
        self.initial_guess.states[2] = numpy.array(
            [self.u0 for _ in range(self.prm['n_nodes'])])

        # Tangential velocity [UA/km]
        self.initial_guess.states[3] = numpy.array(
            [self.v0 for _ in range(self.prm['n_nodes'])])

        # Mass [kg]
        self.initial_guess.states[4] = numpy.linspace(
            self.m0, self.m0 - self.g_tof * self.mass_flow, self.prm['n_nodes'])

        # Controls
        # ========
        self.initial_guess.controls = numpy.ndarray(
            shape=(self.prm['n_controls'], self.prm['n_nodes']))

        # Thrust direction [rad]
        # self.initial_guess.controls[0] = numpy.array(
            # [0 for _ in range(self.prm['n_nodes'])])
        self.initial_guess.controls[0] = numpy.concatenate((self.T*numpy.ones(20), numpy.zeros(370), self.T*numpy.ones(10)))


if __name__ == '__main__':

    # Instantiation of the problem
    problem = PlanarHeliocentricTransfer()

    # Instantiation of the optimization
    optimization = Optimization(problem=problem)

    # Launch of the optimization
    optimization.run()
