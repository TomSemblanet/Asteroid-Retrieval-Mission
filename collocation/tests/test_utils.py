import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre as Legendre_poly
import numpy as np
import unittest
import math

from src.optimal_control import constraints as cs, cost as ct, collocation as cl, pseudospectrals as ps, transcription, problem, utils


class ProblemTest(unittest.TestCase):

    def test_scale_time(self):
        """ Tests the Utils module `scale_time` function """

        # Definition of an input array
        x_in = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Computation of the input array scaled to [-1, 1]
        cpt_x_scl = utils.scale_time(x_in, (-1, 1))
        cor_s_scl = np.array(
            [-1., -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.])

        # Validity check
        self.assertTrue(np.array_equal(cpt_x_scl, cor_s_scl))

        # Computation of the input array scaled to [-10, 0]
        cpt_x_scl = utils.scale_time(x_in, (-10, 0))
        cor_s_scl = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0])

        # Validity check
        self.assertTrue(np.array_equal(cpt_x_scl, cor_s_scl))

    def test_make_decision_variable_vector(self):
        """ Tests the Utils module `make_decision_variable_vector` function """

        # Definition of virtual problem to test the function
        prm = {'n_nodes': 5, 'tr_method_nm': 'trapezoidal'}
        states = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        controls = np.array([[-4, -3, -2, -1, 0]])
        controls_col = np.empty(0)
        f_prm = np.array([0.5, 7])
        t_i, t_f = 0, 10

        # Computation of the decision variables vector
        cpt_dvv = utils.make_decision_variable_vector(
            states, controls, controls_col, t_i, t_f, f_prm, prm)
        cor_dvv = np.array(
            [0, 5, 1, 6, 2, 7, 3, 8, 4, 9, -4, -3, -2, -1, 0, 0.5, 7, 0, 10])

        # Equality check
        self.assertTrue(np.array_equal(cpt_dvv, cor_dvv))

    def test_unpack_decision_variable_vector(self):
        """ Tests the Utils module `unpack_decision_variable_vector` function """

        # Definition of a virtual problem to test the function
        prm = {'n_nodes': 5, 'n_states': 2, 'n_f_par': 2, 'n_controls': 1,
               'n_controls_col': 0, 'tr_method_nm': 'trapezoidal'}
        dvv = np.array([0, 5, 1, 6, 2, 7, 3, 8, 4,
                        9, -4, -3, -2, -1, 0, 0.5, 7, 0, 10])

        # Computation of the states, controls matrices and
        # initial, final times
        cpt_t_i, cpt_t_f, cpt_f_prm, cpt_st, cpt_ctr, cpt_ctr_col = utils.unpack_decision_variable_vector(
            dvv, prm)
        cor_st = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        cor_ctr = np.array([[-4, -3, -2, -1, 0]])
        cor_ctr_col = np.empty(0)
        cor_f_prm = np.array([0.5, 7])
        cor_t_i, cor_t_f = 0, 10

        # Equality check
        self.assertTrue(np.array_equal(cpt_st, cor_st))
        self.assertTrue(np.array_equal(cpt_ctr, cor_ctr))
        self.assertTrue(np.array_equal(cpt_ctr_col, cor_ctr_col))
        self.assertTrue(np.array_equal(cpt_f_prm, cor_f_prm))

        self.assertEqual(cpt_t_i, cor_t_i)
        self.assertEqual(cpt_t_f, cor_t_f)

    def test_retrieve_time_grid(self):
        """ Tests the Utils module `retrive_time_grid` function """

        # Time-steps array
        h_vec = np.array([.2, .2, .2, .2, .2, .2, .2, .2, .2, .2])

        # Initial and final times
        t_i = 0.
        t_f = 10.

        # Computed time grid
        cpt_time_grd = utils.retrieve_time_grid(h_vec, t_i, t_f)
        cpt_time_grd = np.round(cpt_time_grd, 15)

        # Correct time grid
        cor_time_grd = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

        self.assertTrue(np.array_equal(cpt_time_grd, cor_time_grd))
