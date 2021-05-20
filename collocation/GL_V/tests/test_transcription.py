import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre as Legendre_poly
import numpy as np
import cppad_py
import unittest
import math

from src.problem import Problem
from src.transcription import Transcription


class ProblemTest(unittest.TestCase):

    def setUp(self):
        """ Definition of a toy problem """

        # Transcription and optimization options
        options = {'tr_method': 'trapezoidal'}

        # Instantiation of a problem
        self.problem = Problem(n_states=2, n_controls=1, n_f_par=2, n_path_con=0,
                               n_event_con=0, n_nodes=5)

        # Computes the derivatives a state vector
        def dynamics(states, controls, f_par):
            dynamics = np.ndarray(
                (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)

            x1 = states[0]
            x2 = states[1]

            x1_dot = x2
            x2_dot = x1 * x2

            dynamics[0] = x1_dot
            dynamics[1] = x2_dot

            return dynamics

        # Definition of the end-point cost function
        def end_point_cost(ti, xi, tf, xf, f_prm):
            """ Computation of the end-point cost """
            x1_f = xf[0]
            x2_f = xf[1]

            return x2_f

        self.problem.dynamics = dynamics
        self.problem.end_point_cost = end_point_cost

        # Definition of the initial guess
        self.problem.initial_guess.states = np.array(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        self.problem.initial_guess.controls = np.array([[-1, -2, -3, -4, -5]])
        self.problem.initial_guess.f_prm = np.array([0., 1.])
        self.problem.initial_guess.time = np.array([0, 1, 2, 3, 4])

        # Setup of the problem
        self.problem.setup(transcription_method=options['tr_method'])

        # Instantiation of the transcription object
        self.transcription = Transcription(
            problem=self.problem, options=options)

    def test_nodes_adaptation(self):
        """ Test the Transcription class `nodes_adaptation` method """

        # Adaptation of the nodes
        cpt_st_mat, cpt_ct_mat = self.transcription.nodes_adaptation()
        cpt_h_arr, cpt_ti, cpt_tf = self.problem.prm['h'], self.problem.prm['t_i'], self.problem.prm['t_f']

        # Expected results
        cor_st_mat = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        cor_ct_mat = np.array([[-1, -2, -3, -4, -5]])

        cor_h_arr = np.array([.5, .5, .5, .5])
        cor_ti = 0.
        cor_tf = 4.

        # Equality check
        self.assertTrue(np.array_equal(cpt_st_mat, cor_st_mat))
        self.assertTrue(np.array_equal(cpt_ct_mat, cor_ct_mat))
        self.assertTrue(np.array_equal(cpt_h_arr, cor_h_arr))
        self.assertEqual(cpt_ti, cor_ti)
        self.assertEqual(cpt_tf, cor_tf)

    def test_build_decision_variable_vector(self):
        """ Tests the Transcription class `build_decision_variable_vector`	
                method """

        # Recuperation of the states and controls matrices
        states_mat = self.problem.initial_guess.states
        controls_mat = self.problem.initial_guess.controls

        # Computation of the decision variables vector
        cpt_dvv = self.transcription.build_decision_variable_vector(
            states_mat, controls_mat)
        cor_dvv = np.array([0., 5., 1., 6., 2., 7., 3., 8.,
                            4., 9., -1., -2., -3., -4., -5., 0., 1., 0., 4.])

        # Equality check
        self.assertTrue(np.array_equal(cpt_dvv, cor_dvv))

    def test_build_decision_variable_vector_boundaries(self):
        """ Test the Transcription class `build_decision_variable_vector_boundaries`
                method """

        # Setting the lower and upper boundaries for states, controls, free-parameters,
        # initial and final times
        self.problem.low_bnd.states = np.array([0., 1., 2., 3., 4.])
        self.problem.upp_bnd.states = np.array([1., 2., 3., 4., 5.])

        self.problem.low_bnd.controls = np.array([-1, 1])
        self.problem.upp_bnd.controls = np.array([0., 1])

        self.problem.low_bnd.f_par = np.array([0., 1.])
        self.problem.upp_bnd.f_par = np.array([1., 2.])

        self.problem.low_bnd.ti = self.problem.upp_bnd.ti = 0.
        self.problem.low_bnd.tf = self.problem.upp_bnd.tf = 1.

        # Computation of the lower and upper variables boundaries
        cpt_low, cpt_upp = self.transcription.build_decision_variable_vector_boundaries()

        # Excpected lower and upper variables boundaries
        cor_low = np.array([0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4.,
                            -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 0., 1., 0., 1.])
        cor_upp = np.array([1., 2., 3., 4., 5., 1., 2., 3., 4., 5., 1., 2., 3., 4., 5., 1., 2., 3., 4., 5., 1., 2., 3., 4., 5.,
                            0., 1, 0., 1, 0., 1, 0., 1, 0., 1, 1., 2., 0., 1.])

        # Equality check
        self.assertTrue(np.array_equal(cpt_low, cor_low))
        self.assertTrue(np.array_equal(cpt_upp, cor_upp))
