#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 09:50:23 2020

@author: SEMBLANET Tom

"""

import cppad_py
import numpy as np

from src.optimal_control import constraints as cs, cost as ct, \
    collocation as col, pseudospectrals as ps, scaling as sc, utils


class Transcription:
    """ `Transcription` class translates a user-defined optimal control problem into a parametric optimization
        which can be solved by means of Non-Linear Programming Solvers (IPOPT or SNOPT). 

        Parameters
        ----------
        problem : Problem
            Optimal-control problem defined by the user
        options : dict
            Transcription and Optimization options dictionnary

        Attributes
        ----------
        problem : Problem
            Optimal-control problem defined by the user
        decision_variables_vector : array
            Vector of the problem variables (states, controls, free-parameters, time, path constraints and event constraints)
        decision_variables_vector_low : array
            Decision variables vector lower boundaries 
        decision_variables_vector_upp : array
            Decision variables vector upper boundaries 
        constraints : Constraints
            Constraints object managing the constraints computation of the problem
        cost : Cost
            Cost object managing the cost computation of the problem
        scaling : Scaling
            Scaling object managing the computation of the scaling factors
        tr_method : Collocation or Pseudospectral
            Transcription method object used to compute the defects constraints

    """

    def __init__(self, problem, options):
        """ Initialization of the `Transcription` class """

        # User-defined optimal-control problem
        self.problem = problem

        # General options
        self.options = options

        # States, Controls and Time transformation
        states, controls = self.nodes_adaptation()

        # Decision variable vector construction
        self.decision_variables_vector = self.build_decision_variable_vector(states,
                                                                             controls)

        # Decision variables vector boundaries construction
        self.decision_variables_vector_low, self.decision_variables_vector_upp = \
            self.build_decision_variable_vector_boundaries()

        # Constraints object instanciation + constraints
        # boundaries and Jacobian construction
        self.constraints = cs.Constraints(
            problem=self.problem, options=self.options, tr_method=self.tr_method)
        self.constraints.set_up_ad()

        # Cost computation
        self.cost = ct.Cost(problem=self.problem,
                            options=self.options, tr_method=self.tr_method)
        self.cost.set_up_ad()

        # Scaling factors computation
        cost_gradient = self.cost.compute_cost_gradient(
            self.decision_variables_vector)
        self.constraints.compute_constraints_jacobian(
            self.decision_variables_vector)
        self.scaling = sc.Scaling(self.decision_variables_vector_low, self.decision_variables_vector_upp,
                                  self.constraints.jac_dict['jac_data'], cost_gradient, self.problem.prm)

        # Sparse Lagragian Hessian computation
        self.set_up_ad()

    def nodes_adaptation(self):
        """ Scales the time so it belongs to the interval [-1, 1]
            If pseudospectral method is used computation of the LGL and CGL nodes and
            states and controls are interpolated 

            Returns
            -------
            states : ndarray
                Matrix of the states variables
            controls : ndarray
                Matrix of the controls variables 

        """

        # Dictionnary of available transcription methods
        tr_methods_dict = {'trapezoidal': col.Trapezoidal,
                           'hermite-simpson': col.HermiteSimpson,
                           'ps-chebyshev': ps.Chebyshev,
                           'ps-legendre': ps.Legendre}

        x_i = self.problem.initial_guess.states
        u_i = self.problem.initial_guess.controls
        t_i = self.problem.initial_guess.time

        tr_method = tr_methods_dict[self.options['tr_method']]

        # Instanciation and assignment of transcription method to the phase
        self.tr_method = tr_method(self.problem.prm.copy())

        # Computation of new States, Controls, Times according to the
        # choosen transcription method
        states, controls, h = self.tr_method.nodes_adaptation(
            x_i, u_i, t_i)

        # Stocks the information of unscaled t_i, t_f,
        # scaled time-steps and scale factor
        self.problem.prm['h'] = h
        self.problem.prm['t_i'] = t_i[0]
        self.problem.prm['t_f'] = t_i[-1]

        self.problem.prm['sc_factor'] = (t_i[-1] - t_i[0])/2

        return states, controls

    def build_decision_variable_vector(self, states_mat, controls_mat):
        """ Construction of the decision variables vector 

        Parameters
        ----------
        states_mat : ndarray
            Matrix of the states variables
        controls_mat : ndarray
            Matrix of the controls variables

        Returns
        -------
        dvv : array
            Decision variables vector containing the states, controls, free parameters, initial and final time

        """

        # Computation of the phase decision variables vector
        dvv = utils.make_decision_variable_vector(states_mat, controls_mat, self.problem.initial_guess.controls_col,
                                                  self.problem.initial_guess.time[0], self.problem.initial_guess.time[-1],
                                                  self.problem.initial_guess.f_prm, self.problem.prm)

        return dvv

    def build_decision_variable_vector_boundaries(self):
        """ Construction of the decision variables vector lower and upper boundaries 

        Returns
        -------
        low : array
            Decision variables vector lower boundaries
        upp : array
            Decision variables vector upper boundaries 

        """

        # States boundaries initialization
        states_low = np.hstack(
            [self.problem.low_bnd.states] * self.problem.prm['n_nodes'])
        states_upp = np.hstack(
            [self.problem.upp_bnd.states] * self.problem.prm['n_nodes'])

        # Controls boundaries initialization
        controls_low = np.hstack(
            [self.problem.low_bnd.controls] * self.problem.prm['n_nodes'])
        controls_upp = np.hstack(
            [self.problem.upp_bnd.controls] * self.problem.prm['n_nodes'])

        # Controls-mid boundaries initizalization
        controls_col_low = np.hstack(
            [self.problem.low_bnd.controls] * (self.problem.prm['n_nodes']-1)) if self.options['tr_method'] == 'hermite-simpson' else np.empty(0)
        controls_col_upp = np.hstack(
            [self.problem.upp_bnd.controls] * (self.problem.prm['n_nodes']-1)) if self.options['tr_method'] == 'hermite-simpson' else np.empty(0)

        # Concatenation of the states, controls, controls-mid, free-parameters, initial and final time boundaries
        low = np.concatenate((states_low, controls_low, controls_col_low, self.problem.low_bnd.f_par,
                              [self.problem.low_bnd.ti], [self.problem.low_bnd.tf]))
        upp = np.concatenate((states_upp, controls_upp, controls_col_upp, self.problem.upp_bnd.f_par,
                              [self.problem.upp_bnd.ti], [self.problem.upp_bnd.tf]))

        return low, upp

    def set_up_ad(self):
        """ Computation of the Lagrangian mapping function """

        # Computation of the Lagrangian mapping function
        self.mapping_func = self.compute_lagrangian_mapping_function()

        # Creation of a dictionnary containing all the stuff
        # needed to compute the hessian
        self.hess_dict = {}

        # Computation of the constraints hessian
        # sparsity pattern
        self.hess_dict['hess_sp_patt'], self.hess_dict['hess_data'], \
            self.hess_dict['work'] = self.compute_hessian_sparsity_patt()

    def compute_lagrangian_mapping_function(self):
        """ Computes the mapping function between the independants 
            and dependants variables and stores it as an attribut 

            Returns
            -------
            mapping_func : <cppad_py dfun object>
                Mapping function between dependant and independant variables """

        # Decision variables vector used to launch the
        # `compute_cost` and `compute_constraints` methods
        # and record the operations sequence
        ind = np.ones(self.problem.prm['n_var'], dtype=float)
        ind_ = cppad_py.independent(ind)

        # Cost and constraints values returned by the `compute_cost`
        # and  `compute_constraints` methods
        cost = self.cost.compute_cost(ind_)
        cons = self.constraints.compute_constraints(ind_)

        # Array of dependant variables
        dep_ = np.array(np.concatenate((np.array([cost]), cons)))

        # Construction of the mapping function used latter
        # for the computation of the Hessian matrix
        mapping_func = cppad_py.d_fun(ind_, dep_)

        # Optimization of the mapping function
        # (reduction of the number of internal operations)
        mapping_func.optimize()

        return mapping_func

    def compute_hessian_sparsity_patt(self):
        """ Computes the Lagrangian Hessian sparsity pattern 
            and stores it under the form of two numpy arrays containing 
            non-zero elements rows and columns indices 

            Returns
            -------
            hess_sp_patt : <cppad_py sparsity pattern object>
                Hessian sparsity pattern
            hess_data : <cppad_py sparse hessian object>
                Object containing the data about the sparse Hessian such as the row, columns and values 
                of the non-zeros. 
            work : <cppad_py work object> 
                Object used internally by cppad_py. 

        """

        # Hessian number of columns
        n = self.mapping_func.size_domain()
        m = self.mapping_func.size_range()

        # Set up of the matrix pattern
        # (cf. https://bradbell.github.io/cppad_py/doc/xsrst/py_hes_sparsity.html)
        select_d = np.array([True for _ in range(n)])
        select_r = np.array([True for _ in range(m)])

        # Set up of the pattern, just a sparsity
        # pattern that will hold the Hessian sparsity pattern
        hess_sp_patt = cppad_py.sparse_rc()

        # Computation of the Hessian sparsity pattern
        self.mapping_func.for_hes_sparsity(select_d, select_r, hess_sp_patt)

        # Computation of all possibly non-zero entries in Hessian
        hess_data = cppad_py.sparse_rcv()
        hess_data.pat(hess_sp_patt)

        # Work space used internally by cppad
        work = cppad_py.sparse_hes_work()

        return hess_sp_patt, hess_data, work

    def compute_lagrangian_hessian(self, obj_fact, lagrange_mult, decision_variables_vector):
        """ Computes the Hessian of the cost for a given  decision variables vector 

        Parameters
        ----------
        obj_fact : float
            Objective function factor in the computation of the Lagrangian.
        lagrange_mult : array
            Lagrange multipliers (constraints functions factors) in the computation of the Lagrangian.
        decision_variables_vector : array
            Vector of the decision variables. 

        Returns
        -------
        sparse_hess : array
            Array containing the values of the non-zeros elements of the Sparse Hessian 

        """

        # Function multiplier
        r = np.insert(lagrange_mult, 0, obj_fact)

        # Computation of the cost Hessian matrix, the result is
        # stored in the `hess_data` element of `hess_dict`
        self.mapping_func.sparse_hes(self.hess_dict['hess_data'], decision_variables_vector, r,
                                     self.hess_dict['hess_sp_patt'], self.hess_dict['work'])

        # Non-zero sparse hessian value recovery
        sparse_hess = self.hess_dict['hess_data'].val()

        return sparse_hess
