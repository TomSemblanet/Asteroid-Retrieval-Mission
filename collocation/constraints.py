#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 10:30:53 2020

@author: SEMBLANET Tom

"""

import cppad_py
import numpy as np

from src.optimal_control import utils


class Constraints:
    """ `Constraints` class manages the constraints of the problem namely the path, event and
        defects constraints.
        It manages the construction of the constraints lower and upper boundaries vectors,
        the computation of the path, event and defect constraints at each iteration round and the
        computation of the constraints Jacobian using automatic differentiation through `cppad_py`
        library.

        Parameters
        ----------
        problem : Problem
            Optimal-control problem defined by the user
        options : dict
            Transcription and Optimization options dictionnary
        tr_method : Collocation or Pseudospectral
            Transcription method object used to compute the defects constraints

        Attributes
        ----------
        problem : Problem
            Optimal-control problem defined by the user
        low : array
            Constraints vector lower boundaries
        upp : array
            Constraints vector upper boundaries
        mapping_function : <cppad_py dfun object>
            Function mapping the independant and the dependant variables during computation
            of constraints used by automatic differentiation for the construction of the
            constraints Jacobian
        jac_dict : dict
            Dictionnary used to store all the stuff needed to compute and manage the
            constraints Jacobian through cppad_py
        cost : Cost
            Cost object managing the cost computation of the problem
        scaling : Scaling
            Scaling object managing the computation of the scaling factors

    """

    def __init__(self, problem, options, tr_method):
        """ Initialization of the `Constraints` class """

        # User-defined optimal-control problem
        self.problem = problem

        # General options
        self.options = options

        # Transcription method
        self.tr_method = tr_method

        # Construction of the lower and upper constraints boundaries
        self.low, self.upp = self.build_constraints_boundaries()

    def build_constraints_boundaries(self):
        """  Construction of the event, path and defects lower and upper boundaries

        Returns
        -------
        low : array
            Constraints lower boundaries array
        upp : array
            Constraints upper boundaries array

        """

        # Trapezoidal and Hermite-Simpson methods can't compute
        # defects at the last node contrary to pseudospectral methods
        coll_method = self.options['tr_method'] in [
            'trapezoidal', 'hermite-simpson']
        n_nodes = self.problem.prm['n_nodes'] - \
            1 if coll_method else self.problem.prm['n_nodes']

        # Defects lower and upper boundaries
        defects_low = np.zeros(
            self.problem.prm['n_states'] * n_nodes)
        defects_upp = np.zeros(
            self.problem.prm['n_states'] * n_nodes)

        # Path lower and upper boundaries
        path_low = np.hstack([self.problem.low_bnd.path]
                             * (self.problem.prm['n_nodes']))
        path_upp = np.hstack([self.problem.upp_bnd.path]
                             * (self.problem.prm['n_nodes']))

        # Events lower and upper boundaries
        event_low = self.problem.low_bnd.event
        event_upp = self.problem.upp_bnd.event

        # Assembly of the lower and upper boundaries vectors
        low = np.concatenate((defects_low, path_low, event_low))
        upp = np.concatenate((defects_upp, path_upp, event_upp))

        return low, upp

    def compute_constraints(self, decision_variables_vector):
        """ Computation of the path, event and defects constraints.
              Path and Event constraints are computed using user-defined functions while Defects
              constraints are computed through transcription method's intern functions.
              (see `Trapezoidal`, `HermiteSimpson` or `Pseudospectral` classes).


        Parameters
        ----------
        decision_variables_vector : array
           Decision variables vector

        Returns
        -------
        con : array
            Constraints vector containing path, events and defects constraints.

        """

        # Unpacking the decision variables vector
        t_i, t_f, f_prm, states, controls, controls_mid = utils.unpack_decision_variable_vector(
            decision_variables_vector, self.problem.prm)

        # Update of the initial and final times as well as time scaling factor
        self.problem.prm['t_i'] = t_i
        self.problem.prm['t_f'] = t_f
        self.problem.prm['sc_factor'] = (t_f - t_i)/2

        # Computation of the defects constraints and conversio into a 1D-array
        if self.options['tr_method'] == 'hermite-simpson':
            defects_matrix = self.tr_method.compute_defects(
                states, controls, controls_mid, f_prm, self.problem.dynamics,
                self.problem.prm['sc_factor'])
        else:
            defects_matrix = self.tr_method.compute_defects(
                states, controls, f_prm, self.problem.dynamics, self.problem.prm['sc_factor'])
        defects = defects_matrix.flatten(order='F')

        # Computation of path constraints and conversion into a 1D-array
        if self.problem.prm['n_path_con'] != 0:
            path = self.problem.path_constraints(
                states, controls, f_prm).flatten(order='F')
        else:
            path = np.empty(0)

        # Computation of the events constraints and conversion into a 1D-array
        if self.problem.prm['n_event_con'] != 0:
            event = self.problem.event_constraints(states[:, 0], controls[:, 0], states[:, -1],
                                                   controls[:, -1], f_prm, t_i, t_f).flatten()
        else:
            event = np.empty(0)

        # Assembly of the constraints
        con = np.concatenate((defects, path, event))

        return con

    def set_up_ad(self):
        """ Computes the mapping function and the sparsity
            pattern of the constraints Jacobian """

        # Computation of the constraints mapping function
        self.mapping_function = self.compute_mapping_function()

        # Creation of a dictionnary containing all the stuff
        # needed to compute the jacobian
        self.jac_dict = {}

        # Computation of the constraints jacobian
        # sparsity pattern
        self.jac_dict['jac_sp_patt'], self.jac_dict['jac_data'], \
            self.jac_dict['work'] = self.compute_jacobian_sparsity_patt()

    def compute_mapping_function(self):
        """ Computes the mapping function between the independant
              and dependants variables and stores it as an attribut


        Returns
        -------
        mapping_function : <cppad_py dfun object>
            Mapping function between dependant and independant variables. """

        # Decision variables vector used to launch the `compute_constraints`
        # method and records the operations sequence
        ind = np.ones(self.problem.prm['n_var'], dtype=float)
        ind_ = cppad_py.independent(ind)

        # Constraints values returned by the `compute_constraints` method
        dep_ = self.compute_constraints(ind_)

        # Construction of the mapping function used latter for the computation
        # of the Jacobian matrix and its sparsity pattern
        mapping_function = cppad_py.d_fun(ind_, dep_)

        # Opitimization of the mapping function reduces the internal
        # number of variables and operations and so the computation time
        mapping_function.optimize()

        return mapping_function

    def compute_jacobian_sparsity_patt(self):
        """ Computes the cost Hessian sparsity pattern
             and stores it under the form of two numpy arrays containing
             non-zero elements rows and columns indices

            Returns
            -------
            jac_sp_patt : <cppad_py sparsity pattern object>
                Jacobian sparsity pattern
            jac_data : <cppad_py sparse jacobian object>
                Object containing the data about the sparse Jacobian such as the row,
                columns and values
                of the non-zeros.
            work : <cppad_py work object>
                Object used internally by cppad_py.

        """

        # Jacobian number of columns
        n_col = self.mapping_function.size_domain()

        # Set up of the identity matrix pattern
        pattern_in = cppad_py.sparse_rc()
        pattern_in.resize(n_col, n_col, n_col)
        for i in range(n_col):
            pattern_in.put(i, i, i)

        # Sets up of the pattern, just a sparsity
        # pattern that will hold the Jacobian sparsity pattern
        jac_sp_patt = cppad_py.sparse_rc()

        # Computation of the jacobian sparsity pattern
        self.mapping_function.for_jac_sparsity(pattern_in, jac_sp_patt)

        # Computation of all possibly non-zero entries in Jacobian
        jac_data = cppad_py.sparse_rcv()
        jac_data.pat(jac_sp_patt)

        # Work space used to save time for multiple calls
        work = cppad_py.sparse_jac_work()

        return jac_sp_patt, jac_data, work

    def compute_constraints_jacobian(self, decision_variables_vector):
        """ Computes the Jacobian of the constraints for a given
            decision variables vector

        Parameters
        ----------
        decision_variables_vector : array
            Vector of the decision variables

        Returns
        -------
        sparse_jac : array
             Array containing the values of the non-zeros elements of the Sparse Jacobian

        Note
        ----
        In theory, reverse mode should be used for the computation of the Jacobian as the
        number of constraints is lower than the number of variables. However, cppad_py doesn't
        seem to handle it well and forward mode is faster.

        """

        # Computation of the constraints Jacobian matrix,
        # the result is stored in the `jac_data` element of jac_dict
        self.mapping_function.sparse_jac_for(self.jac_dict['jac_data'], decision_variables_vector,
                                             self.jac_dict['jac_sp_patt'], self.jac_dict['work'])

        sparse_jac = self.jac_dict['jac_data'].val()

        return sparse_jac
