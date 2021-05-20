""" Makes the interface between the transcription
	class and the NLP solver (either IPOPT or SNOPT) """

import ipyopt

import numpy as np
import pickle
import math
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from collocation import constraints as cs, cost as ct, collocation as col, pseudospectrals as ps, utils


class IPOPT:
    """ Class `IPOPT` uses the IPyOPT python library to solve NLP problems through IPOPT 
        (for Interior Point OPTimizer) a library for large scale nonlinear optimization of continuous systems.

        Parameters
        ----------
        transcription : Transcription
            Optimal controls transcribed

        Attributs
        ---------
        transcription : Transcription
            Optimal controls transcribed 
        nlp : <IPyOPT Problem object>
            NLP solver 

    """

    def __init__(self, transcription):
        """ Initialization of the `IPOPT` class """
        self.transcription = transcription

    def launch(self):
        """ Launching of the optimization process using IPOPT solver 

            Returns
            -------
            opt_states : ndarray
                Matrix of the states returned by IPOPT
            opt_controls : ndarray
                Matrix of the controls returned by IPOPT
            opt_controls_col : ndarray
                Matrix of the collocation points controls return by IPOPT
            f_prm : array
                Array of the free parameters returned by IPOPT
            time_grid : array
                Array of the time nodes

        """

        problem = self.transcription.problem
        tr_method = self.transcription.tr_method
        tr_method_nm = self.transcription.options['tr_method']

        # Variables lower boundaries
        x_L = self.transcription.decision_variables_vector_low

        # Variables upper boundaries
        x_U = self.transcription.decision_variables_vector_upp

        # Constraints lower boundaries
        g_L = self.transcription.constraints.low

        # Constraints upper boundaries
        g_U = self.transcription.constraints.upp

        # Jacobian sparsity pattern
        jac_nnz = (self.transcription.constraints.jac_dict['jac_data'].row(),
                   self.transcription.constraints.jac_dict['jac_data'].col())

        # Hessian sparsity pattern
        hes_nnz = (self.transcription.hess_dict['hess_data'].row(),
                   self.transcription.hess_dict['hess_data'].col())

        # Set-up the problem
        self.nlp = ipyopt.Problem(problem.prm['n_var'], x_L, x_U, problem.prm['n_con'], g_L, g_U, jac_nnz,
                                  hes_nnz, self.eval_f, self.eval_grad_f, self.eval_g, self.eval_jac_g, self.eval_h)

        # Sends scaling factors to the IPyOPT library
        self.nlp.set_problem_scaling(self.transcription.scaling.obj_fac, self.transcription.scaling.var_fac,
                                     self.transcription.scaling.con_fac)

        # Setting of the IPOPT options
        self.set_IPOPT_options()

        # Definition of solver starting point
        x0 = self.transcription.decision_variables_vector

        # Calling the solver ...
        _x, obj, status = self.nlp.solve(x0)

        # Recuperation of the optimal variables
        t_i, t_f, f_prm, opt_states, opt_controls, opt_controls_col = utils.unpack_decision_variable_vector(
            _x, problem.prm)

        # Reconstruction of the time grid
        time_grid = utils.retrieve_time_grid(problem.prm['h'], t_i, t_f)

        return opt_states, opt_controls, opt_controls_col, f_prm, time_grid

    def eval_f(self, x):
        """ Evaluation of the cost

            Parameters
            ----------
            x : array
                Decision variables vector

            Returns
            -------
            float
                Cost value

         """
        return self.transcription.cost.compute_cost(x)

    def eval_grad_f(self, x, out):
        """ Evaluation of the cost function gradient

            Parameters
            ----------
            x : array
                Decision variables vector
            out : array
                Array created by IPyOPT library where returned values 
                must be stored

        """
        retour = self.transcription.cost.compute_cost_gradient(x)
        for k, v in enumerate(retour):
            out[k] = v

    def eval_g(self, x, out):
        """ Evaluation of the constraints

            Parameters
            ----------
            x : array
                Decision variables vector
            out : array
                Array created by IPyOPT library where returned values 
                have to be stored

        """
        retour = self.transcription.constraints.compute_constraints(x)
        for k, v in enumerate(retour):
            out[k] = v

    def eval_jac_g(self, x, out):
        """ Evaluation of the constraints Jacobian

            Parameters
            ----------
            x : array
                Decision variables vector
            out : array
                Array created by IPyOPT library where returned values 
                must be stored

        """
        jac = self.transcription.constraints.compute_constraints_jacobian(x)
        for k, v in enumerate(jac):
            out[k] = v

    def eval_h(self, x, lagrange_mult, obj_factor, out):
        """ Evaluate the Lagrangian's hessian 

            Parameters
            ----------
            x : array
                Decision variables vector
            obj_factor : float
                Objective function factor in  the computation of the Lagrangian .
            lagrange_mult : array
                Lagrange multipliers (constraints functions factors) in the computation of the Lagrangian.
            out : array
                Array created by IPyOPT library where returned values 
                must be stored

        """
        hes = self.transcription.compute_lagrangian_hessian(
            obj_factor, lagrange_mult, x)
        for k, v in enumerate(hes):
            out[k] = v

    def set_IPOPT_options(self):
        """ Setting of the IPOPT options """

        # Maximum number of iterations
        self.nlp.set(max_iter=500)

        # Uses the linear solver defined by the the user (available solvers are : mumps,
        # ma27, ma57, ma77, ma86)
        self.nlp.set(linear_solver=self.transcription.options['linear_solver'])

        # Scaling factors are automatically computed by the `Scaling` class
        self.nlp.set(nlp_scaling_method='user-scaling')
