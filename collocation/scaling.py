#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 10:30:53 2020

@author: SEMBLANET Tom

"""

import math
import numpy as np

class Scaling:
    """  `Scaling` class computes the scaling factors of decision variables, objective function,
          defects constraints and constraints Jacobian.

            Parameters
            ----------
            dvv_low : array
               Decision variables lower boundaries vector
             dvv_upp : array
               Decision variables upper boundaries vector
             jac : <cppad_py sparse jacobian object>
                 Sparse Jacobian of the constraints function
            grad : <cppad_py gradient object>
                 Gradient of the cost function
            pb_prm : dict
                 Dictionnary containing the problem parameters

            Attributes
            ----------
            pb_prm : dict
                 Dictionnary containing the problem parameters
            var_fac : array
                Decision variables scaling factors
            obj_fac : float
                Objective function scaling factor
            con_fac : array
                Scaling factors of the path, event and defect constraints

        """

    def __init__(self, dvv_low, dvv_upp, jac, grad, pb_prm):
        """ Initialiation of the `Scaling` class """

        # Problem parameters
        self.pb_prm = pb_prm

        # Array of scale factors of the variables
        self.var_fac = self.compute_variables_factors(dvv_low, dvv_upp)

        # Scale factor of the objective function
        self.obj_fac = self.compute_objective_factor(grad)

        # Array of scale factors of the constraints
        self.con_fac = self.compute_constraints_factors(jac)

    @staticmethod
    def compute_variables_factors(dvv_low, dvv_upp):
        """

        Computation of the variables scale factors array


        Parameters
        ----------
        dvv_low : array
               Decision variables lower boundaries vector
       dvv_upp : array
               Decision variables upper boundaries vector

        Returns
        -------
        var_fac : array
            Decision variables scaling factors

        """

        # Array of factors
        var_fac = np.zeros(len(dvv_low))

        for i, (v_low, v_upp) in enumerate(zip(dvv_low, dvv_upp)):
            fact = max(abs(v_low), abs(v_upp))
            if fact != 0:
                var_fac[i] = 1. / max(abs(v_low), abs(v_upp))
            else:
                var_fac[i] = 1.

        return var_fac

    @staticmethod
    def compute_objective_factor(grad):
        """

        Computation of the objective function scaling factor.


        Parameters
        ----------
        grad : <cppad_py gradient object>
            Gradient of the cost function, generated throught cppad_py library.

        Returns
        -------
        fact : float
            Objective function scaling factor

        """

        # Gradient norm
        grad_norm = np.linalg.norm(grad)

        # Factor
        if grad_norm != 0:
            fact = 1./grad_norm
        else:
            fact = 1.

        return fact

    def compute_defects_factors(self):
        """

        Computes the array of defects factors, setted equal to the associtaed variable
                scaling factor.


        Returns
        -------
        defects_fac : array
            Defects scaling factors

        """

        # Number of defects
        coll_method = self.pb_prm['tr_method_nm'] in [
            'trapezoidal', 'hermite-simpson']
        n_defects = (self.pb_prm['n_nodes']-1)*self.pb_prm['n_states'] \
            if coll_method else self.pb_prm['n_nodes']*self.pb_prm['n_states']

        defects_fac = self.var_fac[:n_defects]

        return defects_fac

    def compute_constraints_factors(self, jac_data):
        """

        Computation of the constraints scale factors.


        Parameters
        ----------
        jac_data : array
            Values of the constraints jacobian non-zeros.

        Returns
        -------
        con_fac : array
            Array containing the constraints scaling factors.

        """

        # Computation of defects scaling factors
        defects_fact = self.compute_defects_factors()
        n_defects = len(defects_fact)

        # Array of constraints factors
        n_con = len(np.unique(jac_data.row()))
        con_fac = np.zeros(n_con)

        # Assignation of defects constraints to the constraints
        # factors array
        con_fac[:n_defects] = defects_fact

        jac_val = jac_data.val()
        # Computation of the norm of the Jacobian rows
        for k, row in enumerate(jac_data.row()):
            if row >= n_defects:
                con_fac[row] += jac_val[k]*jac_val[k]

        # Computation of the elements of the matrix
        for i in range(n_defects, len(con_fac)):
            if con_fac[i] == 0:
                con_fac[i] = 1
            else:
                con_fac[i] = 1./math.sqrt(con_fac[i])

        return con_fac
