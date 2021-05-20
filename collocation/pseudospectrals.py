#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 10:30:54 2020

@author: SEMBLANET Tom

"""

import math
import cppad_py
import numpy as np
from scipy import interpolate

from numpy.polynomial.legendre import Legendre as Legendre_poly
from numpy.polynomial.chebyshev import Chebyshev as Chebyshev_poly

from collocation import utils


class Pseudospectral:
    """ `Pseudospectral` class is the Mother-class of the pseudospectral transcription
        methods : Chebyshev and Legendre.  

        A pseudospectral method is a global form of orthogonal collocation, i.e., in a pseudospectral
        method the state is approximated using a global polynomial and collocation is performed at chosen points. 
        Thus, as opposed to local collocation, where the degree of the polynomial is fixed and the number of segments
        (meshes) is varied, in a pseudospectral method the number of meshes is fixed and the degree of the polynomial is varied.

        cf.  A SURVEY OF NUMERICAL METHODS FOR OPTIMAL CONTROL   by   Anil V. Rao

        Parameters
        ----------
        options : dict
            Dictionnary containing the problem and transcription parameters
        nodes : array
            Discretization nodes, can either be LGL (Legendre Gauss Lobatto) or CGL (Chebyshev Gauss Lobatto)
               nodes following the pseudospectral method used
        weights : array
            Weights used to approximate the value of an integrand using either Legendre or Chebyshev quadrature
        D : ndarray
            Differentiation matrix associated to a pseudospectral method (either Chebyshev or Legendre)

        Attributes
        ----------
        options : dict
            Dictionnary containing the problem and transcription parameters
        nodes : array
            Discretization nodes, can either be LGL (Legendre Gauss Lobatto) or CGL (Chebyshev Gauss Lobatto)
            nodes following the pseudospectral methods used
        weights : array
            Weights used to approximate the value of an integrand using either Legendre or Chebyshev quadrature
        D : ndarray
            Differentiation matrix associated to a pseudospectral method (either Chebyshev or Legendre)
        D_T : ndarray
            Differentiation matrix transposed. Stored as an attribute to avoid its computation at each optimization round
        defects : array
            Defects constraints array

        """

    def __init__(self, options, nodes, weights, D):
        """ Initizalization of the `Pseudospectral` class """

        # Transcription method options
        self.options = options

        # Collocation nodes
        self.nodes = nodes

        # Quadrature weights
        self.weights = weights

        # Differenciation matrix
        self.D = D

        # Transposed differentiation matrix
        self.D_T = np.transpose(self.D)

        self.defects = np.ndarray(
            (self.options['n_states'], self.options['n_nodes']), dtype=cppad_py.a_double)

    def quadrature(self, func_values):
        """
        Approximates the integral of a function f over the interval [-1, 1] using 
        either Chebyshev or Legendre quadrature weights

        Parameters
        ----------
        func_values : array
           Values of the function at either LGL or CLG nodes

        Returns
        -------
        sum_ : float
           Approximation of the function integrand

        """

        sum_ = 0
        for k in range(len(self.weights)):
            sum_ += self.weights[k]*func_values[k]

        return sum_

    def compute_defects(self, states, controls, f_prm, f, sc_fac):
        """
        Computes the defect matrix following either the  Chebyshev or Legendre
        pseudospectral method 

        Parameters
        ----------
        states : ndarray
            Matrix of the states 
        controls : ndarray
            Matrix of the controls
        f_prm : array
            Array of the free parameters
        f : function
            Dynamics functions
        sc_fac : float
            Time scaling factor

        Returns
        -------
        defects : array
            Array of the defects constraints

        """

        # Derivatives computation
        F = sc_fac * f(states, controls, f_prm)

        # Defects computation
        self.defects = np.dot(states, self.D_T) - F

        return self.defects

    def nodes_adaptation(self, x_i, u_i, t_i):
        """
        Transformation of the time from [t_f, t_i] to CGL/LGL nodes  by interpolation of the values
        of the states and controls at new nodes

        Parameters
        ----------
        x_i : ndarray
            Matrix of the states initial guess
        u_i : ndarray
            Matrix of the controls initial guess
        t_i : array
            Array of the time grid initial guess

        Returns
        -------
        x : ndarray
            Matrix of the states at either LGL or CGL nodes
        u : ndarray
            Matrix of the controls at either LGL or CGL nodes
        h : array
            Array of the time-steps

        """

        # Scaled time
        scl_t = utils.scale_time(t_i)

        # Computation of time-steps
        h = np.zeros(self.options['n_nodes']-1)
        for k, (t, t_nxt) in enumerate(zip(self.nodes[:-1], self.nodes[1:])):
            h[k] = t_nxt - t

        # Computation of states and controls at CGL nodes
        f_x = interpolate.interp1d(scl_t, x_i, kind='cubic')
        x = f_x(self.nodes)

        # Computation of the controls at the required nodes
        f_u = interpolate.interp1d(scl_t, u_i, kind='cubic')
        u = f_u(self.nodes)

        return x, u, h


class Chebyshev(Pseudospectral):
    """ `Chebyshev` class inherits from the `Pseudospectral` class. Manages the computation of the CGL 
        (Chebyshev-Gauss-Lobatto) nodes, weights and differentiation matrix coefficients. 
        Methods for interpolation using Chebyshev interpolation are also implemented.

        Parameters
        ----------
        options : dict
           Dictionnary containing the problem and transcription parameters

        Attributes
        ----------
        C : <numpy chebyshev polynomials object>
           Chebyshev polynomials generated by the `Chebyshev_poly` numpy library's class
        C_dot : <numpy chebyshev polynomials object>
            Chebyshev polynomials derivatives generated throught the  `Chebyshev_poly` numpy library's class

    """

    def __init__(self, options):
        """ Initialization of the `Chebyshev` class """

        # Chebyshev polynomial of order n_nodes
        self.C = Chebyshev_poly(np.concatenate(
            (np.zeros(options['n_nodes'] - 1), [1])))

        # Chebyshev polynomial of order n_nodes derivative
        self.C_dot = self.C.deriv()

        # Chebyshev-Gauss-Lobatto nodes
        CGL_nodes = Chebyshev.compute_CGL_nodes(options['n_nodes'])

        # Chebyshev quadrature weights
        weights = Chebyshev.quadrature_weights(CGL_nodes)

        # Differentiation matrix
        D = Chebyshev.differentiation_matrix(CGL_nodes)

        # Calling Pseudospectral mother-class constructor
        Pseudospectral.__init__(self, options, CGL_nodes, weights, D)

    @staticmethod
    def compute_CGL_nodes(n_nodes):
        """
        Computes `n_nodes` Chebyshev-Gauss-Lobato (CGL) nodes following equations given at : 
            [1]_http://www.sam.math.ethz.ch/~joergw/Papers/fejer.pdf

        Parameters
        ----------
        n_nodes : int
           Number of CGL nodes required (equal to the number of nodes defined by the user)

        Returns
        -------
        array
            Chebyshev-Gauss-Lobatto nodes 

        """

        return np.array([-math.cos(k * math.pi/(n_nodes-1)) for k in range(n_nodes)])

    @staticmethod
    def quadrature_weights(CGL_nodes):
        """
        Computes the `n_nodes` first Chebyshev quadrature weights following equations given at : 
            [1]_http://www.sam.math.ethz.ch/~joergw/Papers/fejer.pdf 

        Parameters
        ----------
        CGL_nodes : array
           Chebyshev-Gauss-Lobatto nodes

        Returns
        -------
        w : array
            Chebyshev quadrature weights

        """

        N = len(CGL_nodes) - 1

        theta = np.array([math.pi / N * k_ for k_ in range(N+1)])

        w = np.ones(N+1)
        v = np.ones(N-1)

        if N % 2 == 0:
            w[0] = w[-1] = 1 / (N**2 - 1)
            for j in range(1, int(N/2)):
                v -= 2 / (4 * j**2 - 1) * np.cos(2 * j * theta[1:-1])
            v -= 1 / (4 * (N/2)**2 - 1) * np.cos(2 * (N/2) * theta[1:-1])
            w[1:-1] = 2/N * v

        else:
            w[0] = w[-1] = 1 / (N**2)
            for j in range(1, int(N/2)+1):
                v -= 2 / (4 * j**2 - 1) * np.cos(2 * j * theta[1:-1])
            w[1:-1] = 2/N * v

        return w

    @staticmethod
    def differentiation_matrix(CGL_nodes):
        """
        Computes the differentation matrix for the Chebyshev-based method, following equation given at :
            [2]_https://github.com/PSOPT/psopt/blob/master/doc/PSOPT_Manual_R5.pdf

        Parameters
        ----------
        CGL_nodes : array
            Chebyshev-Gauss-Lobatto nodes

        Returns
        -------
        D : ndarray
            Differentiation matrix

        """

        N = len(CGL_nodes) - 1

        # Initialization of the differenciation matrix
        D = np.ones((N+1, N+1))

        # Array of parameters needed to compute D
        a = np.ones(N+1)
        a[0] = a[-1] = 2

        for k in range(N+1):
            for i in range(N+1):
                if k != i:
                    D[k, i] = a[k]/(2*a[i]) * (-1)**(k+i) / (math.sin((k+i)
                                                                      * math.pi/(2*N))*math.sin((k-i)*math.pi/(2*N)))
                elif(k == i and k >= 1 and k <= N-1):
                    D[k, i] = -CGL_nodes[k]/(2 * (math.sin(k*math.pi/N)**2))
                elif(k == 0 and i == 0):
                    D[k, i] = -(2 * N**2 + 1) / 6
                else:
                    D[k, i] = (2 * N**2 + 1) / 6

        return D

    def interpolate(self, time, states, controls, interp_time):
        """
        Interpolates states and controls at nodes given in `interp_time` 
        note: interp_time elements must belong to the interval [-1, 1]

        Parameters
        ----------
        time : array
           Time grid array
        states : ndarray
            States matrix
        controls : ndarray
           Controls matrix
        interp_time : array
            Value of the nodes to which the states and controls must be interpolated

        Returns
        -------
        states_intrp : ndarray
            Matrix of the interpolated states
        controls_intrp : ndarray
            Matrix of the interpolated controls

        """

        # Scaling of `interp_time`in [-1 1]
        interp_time = utils.scale_time(interp_time, (-1, 1))

        states_intrp = np.zeros((self.options['n_states'], len(interp_time)))
        controls_intrp = np.zeros(
            (self.options['n_controls'], len(interp_time)))

        N = len(self.nodes) - 1

        # Coefficients array
        c_k = np.ones(N+1)
        c_k[0] = c_k[-1] = 2

        for j, t in enumerate(interp_time):
            ind = np.searchsorted(self.nodes, t, side="right") - 1

            if t in self.nodes:
                states_intrp[:, j] = states[:, ind]
                controls_intrp[:, j] = controls[:, ind]

            else:
                # Lagrange interpolating polynomials
                lagrange_poly = [(-1.)**k / (N*N*c_k[k]) * (1-t*t)*self.C_dot(t) /
                                 (t - tau_k) for k, tau_k in enumerate(self.nodes)]

                # States interpolation at node j
                for k, state in enumerate(states):
                    # Lagrange polynomials
                    states_intrp[k, j] = sum(
                        [state[i] * lagrange_poly[i] for i in range(len(self.nodes))])

                # Controls interpolation at node j
                for k, control in enumerate(controls):
                    # Lagrange polynomials
                    controls_intrp[k, j] = sum(
                        [control[i] * lagrange_poly[i] for i in range(len(self.nodes))])

        return states_intrp, controls_intrp

    def interpolate_ctrl(self, time, controls, tau):
        """
        Interpolatation of the controls at a given time 

        Parameters
        ----------
        time : array
           Time grid array
        controls : ndarray
            Matrix of the controls
        controls_mid : ndarray
            Matrix of the mid-controls
        tau : float
            Value of the node to which the controls must be interpolated

        Returns
        -------
        controls_intrp : array
            Value of the controls at the interpolation time `tau`

        """

        tau = utils.scale_time([time[0], tau, time[-1]], (-1, 1))[1]

        controls_intrp = np.zeros(len(controls))

        # CGL nodes are numeroted tau_0, ..., tau_N
        N = len(self.nodes) - 1

        # Coefficients array
        c_k = np.ones(N+1)
        c_k[0] = c_k[-1] = 2

        if tau in self.nodes:
            ind = np.searchsorted(self.nodes, tau, side="right") - 1
            for k, control in enumerate(controls):
                controls_intrp[k] = control[ind]

        else:
            # Lagrange interpolating polynomials
            lagrange_poly = [(-1.)**k / (N*N*c_k[k]) * (1-tau*tau)*self.C_dot(tau) /
                             (tau - tau_k) for k, tau_k in enumerate(self.nodes)]

            for k, control in enumerate(controls):
                controls_intrp[k] = sum(
                    [control[i] * lagrange_poly[i] for i in range(len(self.nodes))], 0)

        return controls_intrp


class Legendre(Pseudospectral):
    """ `Legendre` class inherits from the `Pseudospectral` class. Manages the computation of the LGL 
        (Legendre-Gauss-Lobatto) nodes, weights and differentiation matrix coefficients. 
        Methods for interpolation using Legendre interpolation are also implemented.

         Parameters
        ----------
        options : dict
           Dictionnary containing the problem and transcription parameters

        Attributes
        ----------
        L : <numpy legendre polynomials object>
           Legendre polynomials generated by the `Legendre_poly` numpy library's class
        L_dot : <numpy legendre polynomials object>
            Legendre polynomials derivatives generated throught the  `Legendre_poly` numpy library's class

    """

    def __init__(self, options):
        """ Initialization of the `Legendre` class """

        # Legendre polynomial of order n_nodes
        self.L = Legendre_poly(np.concatenate(
            (np.zeros(options['n_nodes'] - 1), [1])))

        # Legendre polynomial of order n_nodes derivative
        self.L_dot = self.L.deriv()

        # Legendre-Gauss-Lobatto nodes
        LGL_nodes = Legendre.compute_LGL_nodes(self.L, options['n_nodes'])

        # Chebyshev quadrature weights
        weights = Legendre.quadrature_weights(self.L, LGL_nodes)

        # Differentiation matrix
        D = Legendre.differentiation_matrix(self.L, LGL_nodes)

        # Calling Pseudospectral mother-class constructor
        Pseudospectral.__init__(self, options, LGL_nodes, weights, D)

        # Legendre polynomial evaluated at LGL nodes
        self.L_eval = np.array([self.L(tau_k) for tau_k in LGL_nodes])

    @staticmethod
    def compute_LGL_nodes(L, n_nodes):
        """
        Computes `n_nodes` Legendre-Gauss-Lobato (LGL)   nodes following equations given at : 
            [3]_Elnagar, Gamal & Kazemi, Mohammad & Razzaghi, Mohsen. (1995). 
                Pseudospectral Legendre method for discretizing optimal control problems.

        Parameters
        ----------
         L : <numpy legendre polynomials object>
               Legendre polynomials generated by the `Legendre_poly` numpy library's class
        n_nodes : int
           Number of LGL nodes required (equal to the number of nodes defined by the user)

        Returns
        -------
        array
            Legendre-Gauss-Lobatto nodes 

        """

        # Derivative of Legendre poylnomial
        L_dot = Legendre_poly.deriv(L)

        # n_nodes Legendre-Gauss-Lobatto nodes
        LGL_nodes = np.concatenate(([-1], L_dot.roots(), [1]))

        return LGL_nodes

    @staticmethod
    def quadrature_weights(L, LGL_nodes):
        """
        Computes `n_nodes` Legendre quadrature weights following equations given at : 
            [3]_Elnagar, Gamal & Kazemi, Mohammad & Razzaghi, Mohsen. (1995). 
                Pseudospectral Legendre method for discretizing optimal control problems.

        Parameters
        ----------
       L : <numpy legendre polynomials object>
               Legendre polynomials generated by the `Legendre_poly` numpy library's class
        LGL_nodes : array
           Legendre-Gauss-Lobatto nodes

        Returns
        -------
        w : array
            Legendre quadrature weights

        """

        # LGL node are numeroted tau_0, ..., tau_N
        N = len(LGL_nodes) - 1

        # Quadrature weights array
        w = (2/(N*(N+1))) * 1/(L(LGL_nodes)**2)

        return w

    @staticmethod
    def differentiation_matrix(L, LGL_nodes):
        """
        Computes the differentation matrix for the  Legendre-based method, following equation given at :
            [3]_Elnagar, Gamal & Kazemi, Mohammad & Razzaghi, Mohsen. (1995). 
                Pseudospectral Legendre method for discretizing optimal control problems.

        Parameters
        ----------
        L : <numpy legendre polynomials object>
               Legendre polynomials generated by the `Legendre_poly` numpy library's class
        LGL_nodes : array
            Legendre-Gauss-Lobatto nodes

        Returns
        -------
        D : ndarray
            Differentiation matrix

        """

        # LGL nodes are numeroted tau_0, ..., tau_N
        N = len(LGL_nodes) - 1

        # Initialization of the differenciation matrix
        D = np.ones((N+1, N+1))

        # Array of parameters needed to compute D
        for k in range(N+1):
            for i in range(N+1):
                if k != i:
                    D[k, i] = (L(LGL_nodes[k])/L(LGL_nodes[i])) * \
                        1/(LGL_nodes[k]-LGL_nodes[i])
                elif(k == 0 and i == 0):
                    D[k, i] = -N*(N+1)/4
                elif(k == N and i == N):
                    D[k, i] = N*(N+1)/4
                else:
                    D[k, i] = 0

        return D

    def interpolate(self, time, states, controls, interp_time):
        """
        Interpolates states and controls at nodes given in `interp_time` 
        note: interp_time elements must belong to the interval [-1, 1]

        Parameters
        ----------
        time : array
           Time grid array
        states : ndarray
            States matrix
        controls : ndarray
           Controls matrix
        interp_time : array
            Value of the nodes to which the states and controls must be interpolated

        Returns
        -------
        states_intrp : ndarray
            Matrix of the interpolated states
        controls_intrp : ndarray
            Matrix of the interpolated controls

        """

        # Scaling of `interp_time`in [-1 1]
        interp_time = utils.scale_time(interp_time, (-1, 1))

        states_intrp = np.zeros((self.options['n_states'], len(interp_time)))
        controls_intrp = np.zeros(
            (self.options['n_controls'], len(interp_time)))

        # LGL nodes are numeroted tau_0, ..., tau_N
        N = len(self.nodes) - 1

        for j, t in enumerate(interp_time):
            ind = np.searchsorted(self.nodes, t, side="right") - 1

            if t in self.nodes:
                states_intrp[:, j] = states[:, ind]
                controls_intrp[:, j] = controls[:, ind]

            else:
                # Lagrange interpolating polynomials
                lagrange_poly = [1./(N*(N+1)*self.L(tau_k)) * (t*t - 1)
                                 * self.L_dot(t) / (t - tau_k) for tau_k in self.nodes]

                # States interpolation at node j
                for k, state in enumerate(states):
                    # Lagrange polynomials
                    states_intrp[k, j] = sum(
                        [state[i] * lagrange_poly[i] for i in range(len(self.nodes))])

                # Controls interpolation at node j
                for k, control in enumerate(controls):
                    # Lagrange polynomials
                    controls_intrp[k, j] = sum(
                        [control[i] * lagrange_poly[i] for i in range(len(self.nodes))])

        return states_intrp, controls_intrp

    def interpolate_ctrl(self, time, controls, tau):
        """
        Interpolatation of the controls at a given time 

        Parameters
        ----------
        time : array
           Time grid array
        controls : ndarray
            Matrix of the controls
        controls_mid : ndarray
            Matrix of the mid-controls
        tau : float
            Value of the node to which the controls must be interpolated

        Returns
        -------
        controls_intrp : array
            Value of the controls at the interpolation time `tau`

        """

        tau = utils.scale_time([time[0], tau, time[-1]], (-1, 1))[1]

        controls_intrp = np.zeros(len(controls))

        # CGL nodes are numeroted tau_0, ..., tau_N
        N = len(self.nodes) - 1

        # Coefficients array
        c_k = np.ones(N+1)
        c_k[0] = c_k[-1] = 2

        if tau in self.nodes:
            ind = np.searchsorted(self.nodes, tau, side="right") - 1
            for k, control in enumerate(controls):
                controls_intrp[k] = control[ind]

        else:
            # Lagrange interpolating polynomials
            lagrange_poly = [1./(N*(N+1)*self.L_eval[k]) * (tau*tau - 1)
                             * self.L_dot(tau) / (tau - tau_k) for k, tau_k in enumerate(self.nodes)]

            for k, control in enumerate(controls):
                controls_intrp[k] = sum(
                    [control[i] * lagrange_poly[i] for i in range(len(self.nodes))], 0)

        return controls_intrp
