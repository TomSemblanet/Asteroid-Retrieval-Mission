import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre as Legendre_poly
import numpy as np
import unittest
import math

from src.optimal_control import constraints as cs, cost as ct, collocation as cl, pseudospectrals as ps, transcription, problem, utils


class ProblemTest(unittest.TestCase):

    def test_pseudospectral_quadrature(self):
        """ Tests the Pseudospectral class `quadrature` method """

        # Number of CGL / LGL nodes
        N = 100

        # Definition of options dictionnary, necessary
        # to instanciate Chebyshev & Legendre objects
        options = {'n_nodes': N, 'n_states': 1}

        # Instantiation of Chebyshev & Legendre objects
        cheb = ps.Chebyshev(options)
        leg = ps.Legendre(options)

        # Computation of exponential function
        # at CGL / LGL nodes
        cheb_exp = np.exp(cheb.nodes)
        leg_exp = np.exp(leg.nodes)

        # Computation of the integral
        cheb_computed_int = cheb.quadrature(cheb_exp)
        leg_computed_int = leg.quadrature(leg_exp)

        # Correct value of the integral
        correct_int = np.exp(1) - np.exp(-1)

        # Equality verification (we round the results as the
        # quadrature will never match the exact integral value)
        self.assertEqual(round(cheb_computed_int, 5), round(correct_int, 5))
        self.assertEqual(round(leg_computed_int, 5), round(correct_int, 5))

    def test_pseudospectral_compute_defects(self):
        """ Tests the Pseudospectral class `compute_defects` method """

        # Number of CGL / LGL nodes
        N = 50

        # Definition of options dictionnary, necessary
        # to instanciate Chebyshev & Legendre objects
        options = {'n_nodes': N, 'n_states': 1}

        # Instantiation of Chebyshev & Legendre objects
        cheb = ps.Chebyshev(options)
        leg = ps.Legendre(options)

        # Definition of a test function and its analitic derivative,
        # used to verify that the differentiations made by Chebyshev & Legendre
        # methods are corrects
        cheb_function = [np.sin(5*x_*x_) for x_ in cheb.nodes]
        cor_cheb_function_dot = [10*x_*np.cos(5*x_*x_) for x_ in cheb.nodes]

        leg_function = [np.sin(5*x_*x_) for x_ in leg.nodes]
        cor_leg_function_dot = [10*x_*np.cos(5*x_*x_) for x_ in leg.nodes]

        # Derivatives computed via Chebyshev and Legendre differentiation
        # matrices
        cpt_cheb_function_dot = np.dot(cheb_function, np.transpose(cheb.D))
        cpt_leg_function_dot = np.dot(leg_function, np.transpose(leg.D))

        # Equality verification
        self.assertTrue(np.array_equal(
            np.round(cpt_cheb_function_dot, 3), np.round(cor_cheb_function_dot, 3)))
        self.assertTrue(np.array_equal(
            np.round(cpt_leg_function_dot, 3), np.round(cor_leg_function_dot, 3)))

    def test_pseudospectral_nodes_adaptation(self):
        """ Tests the Pseudospectral class `nodes_adaptation` method """

        # Number of nodes
        N = 10

        # Unscaled time
        uscl_time = np.array([0, 2, 5, 6, 7, 12, 13, 20, 22, 25])

        # Test states & controls arrays
        states = np.array([[x**2 for x in uscl_time]])
        controls = np.array([[math.cos(.2 * x) for x in uscl_time]])

        # Unscaled time-steps
        uscl_h = uscl_time[1:] - uscl_time[:-1]

        # Instantiation of Chebyshev & Legendre objects
        options = {'n_nodes': N, 'n_states': 1, 'n_controls': 1}
        cheb = ps.Chebyshev(options)
        leg = ps.Legendre(options)

        # Computation of states, controls and time-steps
        # at CGL nodes
        cheb_st, cheb_con, cheb_h = cheb.nodes_adaptation(
            states, controls, uscl_time)

        # Computation of states, controls and time-steps
        # at LGL nodes
        leg_st, leg_con, leg_h = leg.nodes_adaptation(
            states, controls, uscl_time)

        # Time-steps validity check
        cpt_cheb_h = cheb_h
        cor_cheb_h = cheb.nodes[1:] - cheb.nodes[:-1]

        cpt_leg_h = leg_h
        cor_leg_h = leg.nodes[1:] - leg.nodes[:-1]

        self.assertTrue(np.array_equal(cpt_cheb_h, cor_cheb_h))
        self.assertTrue(np.array_equal(cpt_leg_h, cor_leg_h))

        # States interpolation validity check
        cpt_cheb_st = cheb_st
        cor_cheb_st = np.array(
            [[x**2 for x in utils.scale_time(cheb.nodes, (0, 25))]])

        cpt_leg_st = leg_st
        cor_leg_st = np.array(
            [[x**2 for x in utils.scale_time(leg.nodes, (0, 25))]])

        for cpt, cor in zip(cpt_cheb_st[0], cor_cheb_st[0]):
            self.assertTrue(abs(cpt-cor) < 10)
        for cpt, cor in zip(cpt_leg_st[0], cor_leg_st[0]):
            self.assertTrue(abs(cpt-cor) < 10)

        # Controls interpolation validity check
        cpt_cheb_ctr = cheb_con
        cor_cheb_ctr = np.array(
            [[math.cos(.2 * x) for x in utils.scale_time(cheb.nodes, (0, 25))]])

        cpt_leg_ctr = leg_con
        cor_leg_ctr = np.array([[math.cos(.2 * x)
                                 for x in utils.scale_time(leg.nodes, (0, 25))]])

        for cpt, cor in zip(cpt_cheb_ctr[0], cor_cheb_ctr[0]):
            self.assertTrue(abs(cpt-cor) < 1)
        for cpt, cor in zip(cpt_leg_ctr[0], cor_leg_ctr[0]):
            self.assertTrue(abs(cpt-cor) < 1)

    def test_chebyshev_compute_CGL_nodes(self):
        """ Tests the Chebyshev class `compute_CGL_nodes` method """

        # Number of Chebyshev-Gauss-Lobatto nodes
        N = 10

        # Computation of N CGL nodes
        computed_nodes = ps.Chebyshev.compute_CGL_nodes(N)

        # Correct N CGL nodes computed by chebfun
        # (Matlab module)
        correct_nodes = np.array(
            [-1.0000, -0.9397, -0.7660, -0.5000, -0.1736, 0.1736, 0.5000, 0.7660, 0.9397, 1.0000])

        # Equality verification
        self.assertTrue(np.array_equal(
            np.round(computed_nodes, 3), np.round(correct_nodes, 3)))

    def test_chebyshev_quadrature_weights(self):
        """ Tests the Chebyshev class `quadrature_weights` method """

        # Number of Chebyshev quadrature weights
        N = 10

        # Computation of N CGL nodes
        CGL_nodes = ps.Chebyshev.compute_CGL_nodes(N)

        # Computation of N Chebyshev quadrature weights
        computed_weights = ps.Chebyshev.quadrature_weights(CGL_nodes)

        # Correct N Chebyshev qaudrature weights computed by chebfun
        # (Matlab module)
        correct_weights = np.array(
            [0.0123, 0.1166, 0.2253, 0.3019, 0.3439, 0.3439, 0.3019, 0.2253, 0.1166, 0.0123])

        # Equality verification
        self.assertTrue(np.array_equal(
            np.round(computed_weights, 3), np.round(correct_weights, 3)))

    def test_chebyshev_differentiation_matrix(self):
        """ Tests the Chebyshev class `differentiation_matrix` method """

        # Dimension of the differentiation matrix
        N = 10

        # Computation of N CGL nodes
        CGL_nodes = ps.Chebyshev.compute_CGL_nodes(N)

        # Computation of N-dim Chebyshev differentiation matrix
        computed_matrix = ps.Chebyshev.differentiation_matrix(CGL_nodes)

        # Correct N-dim Chebyshev differentiation matrix
        # Computed by chebfun (Matlab module)
        correct_matrix = np.array([[-27.1667, 33.1634, -8.5486, 4.0000, -2.4203, 1.7041, -1.3333, 1.1325, -1.0311, 0.5000],
                                   [-8.2909, 4.0165, 5.7588, -2.2743, 1.3054, -
                                       0.8982, 0.6946, -0.5863, 0.5321, -0.2578],
                                   [2.1372, -5.7588, 0.9270, 3.7588, -1.6881,
                                       1.0642, -0.7899, 0.6527, -0.5863, 0.2831],
                                   [-1.0000, 2.2743, -3.7588, 0.3333, 3.0642, -
                                       1.4845, 1.0000, -0.7899, 0.6946, -0.3333],
                                   [0.6051, -1.3054, 1.6881, -3.0642, 0.0895,
                                       2.8794, -1.4845, 1.0642, -0.8982, 0.4260],
                                   [-0.4260, 0.8982, -1.0642, 1.4845, -2.8794, -
                                       0.0895, 3.0642, -1.6881, 1.3054, -0.6051],
                                   [0.3333, -0.6946, 0.7899, -1.0000, 1.4845, -
                                       3.0642, -0.3333, 3.7588, -2.2743, 1.0000],
                                   [-0.2831, 0.5863, -0.6527, 0.7899, -1.0642,
                                       1.6881, -3.7588, -0.9270, 5.7588, -2.1372],
                                   [0.2578, -0.5321, 0.5863, -0.6946, 0.8982, -
                                       1.3054, 2.2743, -5.7588, -4.0165, 8.2909],
                                   [-0.5000, 1.0311, -1.1325, 1.3333, -1.7041, 2.4203, -4.0000, 8.5486, -33.1634, 27.1667]])

        # Equality verification
        for cpt_row, cor_row in zip(computed_matrix, correct_matrix):
            self.assertTrue(np.array_equal(
                np.round(cpt_row, 4), np.round(cor_row, 4)))

    def test_legendre_compute_LGL_nodes(self):
        """ Tests the Legendre class `compute_LGL_nodes` method """

        # Number of Legendre-Gauss-Lobatto nodes
        N = 10

        # Legendre polynomial of order N
        L = Legendre_poly(np.concatenate((np.zeros(N - 1), [1])))

        # Computation of N LGL nodes
        computed_nodes = ps.Legendre.compute_LGL_nodes(L, N)

        # Correct N LGL nodes computed by lepoly
        # (Matlab module)
        correct_nodes = np.array(
            [-1.0000, -0.9195, -0.7388, -0.4779, -0.1653, 0.1653, 0.4779, 0.7388, 0.9195, 1.0000])

        # Equality verification
        self.assertTrue(np.array_equal(
            np.round(computed_nodes, 3), np.round(correct_nodes, 3)))

    def test_legendre_quadrature_weights(self):
        """ Tests the Legendre class `quadrature_weights` method """

        # Number of Legendre quadrature weights
        N = 10

        # Legendre polynomial of order N
        L = Legendre_poly(np.concatenate((np.zeros(N - 1), [1])))

        # Computation of N LGL nodes
        LGL_nodes = ps.Legendre.compute_LGL_nodes(L, N)

        # Computation of N Chebyshev quadrature weights
        computed_weights = ps.Legendre.quadrature_weights(L, LGL_nodes)

        # Correct N Legendre qaudrature weights computed by lepoly
        # (Matlab module)
        correct_weights = np.array(
            [0.0222, 0.1333, 0.2249, 0.2920, 0.3275, 0.3275, 0.2920, 0.2249, 0.1333, 0.0222])

        # Equality verification
        self.assertTrue(np.array_equal(
            np.round(computed_weights, 3), np.round(correct_weights, 3)))

    def test_legendre_differentiation_matrix(self):
        """ Tests the Legendre class `differentiation_matrix` method """

        # Dimension of the differentiation matrix
        N = 10

        # Legendre polynomial of order N
        L = Legendre_poly(np.concatenate((np.zeros(N - 1), [1])))

        # Computation of N LGL nodes
        LGL_nodes = ps.Legendre.compute_LGL_nodes(L, N)

        # Computation of N-dim Legendre differentiation matrix
        computed_matrix = ps.Legendre.differentiation_matrix(L, LGL_nodes)

        # Correct N-dim Chebyshev differentiation matrix
        # Computed by chebfun (Matlab module)
        correct_matrix = np.array([[-22.5000, 30.4381, -12.1779, 6.9438, -4.5994, 3.2946, -2.4529, 1.8296, -1.2760, 0.5000],
                                   [-5.0741, 0, 7.1855, -3.3517, 2.0782, -
                                       1.4449, 1.0592, -0.7832, 0.5438, -0.2127],
                                   [1.2034, -4.2593, 0, 4.3687, -2.1044,
                                       1.3349, -0.9366, 0.6768, -0.4643, 0.1808],
                                   [-0.5284, 1.5299, -3.3641, 0, 3.3873, -
                                       1.6465, 1.0462, -0.7212, 0.4835, -0.1866],
                                   [0.3120, -0.8458, 1.4449, -3.0202, 0,
                                       3.0252, -1.4681, 0.9166, -0.5881, 0.2235],
                                   [-0.2235, 0.5881, -0.9166, 1.4681, -3.0252,
                                       0, 3.0202, -1.4449, 0.8458, -0.3120],
                                   [0.1866, -0.4835, 0.7212, -1.0462, 1.6465, -
                                       3.3873, 0, 3.3641, -1.5299, 0.5284],
                                   [-0.1808, 0.4643, -0.6768, 0.9366, -1.3349,
                                       2.1044, -4.3687, 0.0000, 4.2593, -1.2034],
                                   [0.2127, -0.5438, 0.7832, -1.0592, 1.4449, -
                                       2.0782, 3.3517, -7.1855, 0.0000, 5.0741],
                                   [-0.5000, 1.2760, -1.8296, 2.4529, -3.2946, 4.5994, -6.9438, 12.1779, -30.4381, 22.5000]])

        # Equality verification
        for cpt_row, cor_row in zip(computed_matrix, correct_matrix):
            self.assertTrue(np.array_equal(
                np.round(cpt_row, 4), np.round(cor_row, 4)))
