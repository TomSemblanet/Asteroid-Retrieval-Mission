import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre as Legendre_poly
import numpy as np
import unittest
import math

from src.optimal_control import constraints as cs, cost, collocation as cl, pseudospectrals as ps, transcription, problem, utils


class ProblemTest(unittest.TestCase):

    def setUp(self):
        """ Definition of a toy-problem to test the Cost
                class methods """

        # Definition of the integrand cost function
        def integrand_cost(states, controls, f_prm):
            """ Computation of the toy-problem integrand cost """
            x1 = states[0]
            x2 = states[1]

            return [x1_*x1_ + x2_*x2_ for x1_, x2_ in zip(x1, x2)]

        # Definition of the end-point cost function
        def end_point_cost(ti, xi, tf, xf, f_prm):
            """ Computation of the end-point cost """
            x1_f = xf[0]
            x2_f = xf[1]

            return x2_f

        # Definition of the unique phase of the toy-problem
        self.problem = problem.Problem(n_states=2, n_controls=1, n_f_par=2, n_path_con=0,
                                       n_event_con=0, n_nodes=5)

        # Assignation of the cost functions
        self.problem.integrand_cost = integrand_cost
        self.problem.end_point_cost = end_point_cost

        # Setup of the problem
        self.problem.setup(transcription_method='trapezoidal')

        # Definition of the problem parameters
        pb_prm = self.problem.prm
        pb_prm['h'] = [.5, .5, .5, .5]
        pb_prm['t_i'] = 0
        pb_prm['t_f'] = 10
        pb_prm['sc_factor'] = 5

        pb_prm['tr_method'] = 'trapezoidal'

        # Transcription method instantiation
        tr_method = cl.Trapezoidal(pb_prm.copy())

        self.cost_obj = cost.Cost(self.problem, pb_prm, tr_method)

    def test_compute_cost(self):
        """ Tests the Cost class `compute_cost` method """

        # Definition of a test decision variables vector
        states = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
        controls = np.array([[0, 0, 0, 0, 0, ]])
        controls_col = np.empty(0)
        f_prm = np.array([1., 2.])
        t_i, t_f = 0, 10

        dvv = utils.make_decision_variable_vector(
            states, controls, controls_col, t_i, t_f, f_prm, self.cost_obj.options)

        # Cost computation
        cpt_cost = self.cost_obj.compute_cost(dvv)
        cor_cost = 52.0

        # Equality verification
        self.assertEqual(cpt_cost, cor_cost)
