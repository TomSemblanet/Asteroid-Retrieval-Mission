import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre as Legendre_poly
import numpy as np
import unittest
import math

from src import constraints, cost as ct, collocation as cl, pseudospectrals as ps, transcription, problem, utils


class ProblemTest(unittest.TestCase):

    def setUp(self):
        """ Definition of a toy-problem to test the Constraints
        class methods """

        n_states = 2
        n_controls = 1
        n_f_par = 2
        n_nodes = 3
        n_path_constraints = 2
        n_event_constraints = 2

        # Definition of the path constraints function
        def path_constraints(states, controls, f_par):
            """ Computation of the toy-problem path constraints """
            paths = np.ndarray(
                (n_path_constraints, n_nodes), dtype=float)

            x1 = states[0]
            x2 = states[1]

            paths[0] = [x1_*x1_ + x2_*x2_ for x1_, x2_ in zip(x1, x2)]
            paths[1] = [x1_ + x2_ for x1_, x2_ in zip(x1, x2)]

            return paths

        def event_constraints(xi, ui, xf, uf, ti, tf, f_par):
            """ Computation of the toy-problem event constraints """
            events = np.ndarray((n_event_constraints, 1),
                                dtype=float)

            x1_i, x1_f = xi[0], xf[0]
            x2_i, x2_f = xi[1], xf[1]

            events[0] = x1_f - x1_i
            events[1] = x2_f - x2_i

            return events

        # Definition of the unique phase of the toy-problem
        self.problem = problem.Problem(n_states=n_states, n_controls=n_controls, n_f_par=n_f_par, n_path_con=n_path_constraints,
                                       n_event_con=n_event_constraints, n_nodes=n_nodes)

        # Assignation of the constraints functions
        self.problem.path_constraints = path_constraints
        self.problem.event_constraints = event_constraints

        # Setup of the problem
        self.problem.setup(transcription_method='trapezoidal')

        # Definition of the problem parameters
        pb_prm = self.problem.prm
        pb_prm['h'] = np.array([.5, .5])
        pb_prm['t_i'] = 0
        pb_prm['t_f'] = 10
        pb_prm['sc_factor'] = 5
        pb_prm['tr_method'] = 'trapezoidal'

        tr_method = cl.Trapezoidal(pb_prm.copy())

        # Definition of constraints lower and upper boundaries
        self.problem.low_bnd.path[0] = self.problem.upp_bnd.path[0] = 1

        self.problem.low_bnd.path[1] = -1
        self.problem.upp_bnd.path[1] = 1

        self.problem.low_bnd.event[0] = self.problem.upp_bnd.event[0] = 20
        self.problem.low_bnd.event[1] = self.problem.upp_bnd.event[1] = 10

        # Instantiation of the toy-problem
        self.constr_obj = constraints.Constraints(
            problem=self.problem, options=pb_prm, tr_method=tr_method)

    def test_build_constraints_boundaries(self):
        """ Tests the Constraints class `build_constraints_boundaries` method """

        # Computation of constraints lower and upper boundaries
        cpt_low, cpt_upp = self.constr_obj.build_constraints_boundaries()

        # Correct constraints lower and upper boundaries
        cor_low = np.array(
            [0., 0., 0., 0., 1., -1., 1., -1., 1., -1., 20., 10.])
        cor_upp = np.array(
            [0., 0., 0., 0., 1.,  1., 1.,  1., 1.,  1., 20., 10.])

        # Equality check
        self.assertTrue(np.array_equal(cpt_low, cor_low))
        self.assertTrue(np.array_equal(cpt_upp, cor_upp))

    def test_compute_constraints(self):
        """ Tests the Constraints class `compute_constraints` method """

        # Definition of dynamics function used to compute defects
        # (just the derivatives of the states)
        def dynamics(states, controls, f_par):
            """ Dynamics function """
            dynamics = np.ndarray(
                (states.shape[0], states.shape[1]), dtype=float)

            dynamics[0] = -np.sin(states[0])
            dynamics[1] = np.cos(states[1])

            return dynamics

        # Assignation of the dynamics function to the unique phase
        self.problem.dynamics = dynamics

        # Definition of states, controls and dynamics matrices
        states = np.array([np.cos(np.array([-1, 0., 1.])),
                           np.sin(np.array([-1., 0., 1.]))])
        controls = np.array([[0., 0., 0.]])
        controls_col = np.empty(0)
        f_par = np.array([1., 2.])

        decision_variables_vector = utils.make_decision_variable_vector(
            states, controls, controls_col, 0., 10., f_par, self.constr_obj.options)

        # Computation of the constraints vector
        cpt_con = self.constr_obj.compute_constraints(
            decision_variables_vector)

        # Correct constraints vector
        cor_con = np.array([2.155, -1.241, 1.235, -1.241,
                            1., -0.301, 1., 1., 1., 1.382, 0., 1.683])

        # Equality check
        self.assertTrue(np.array_equal(
            np.round(cpt_con, 3), np.round(cor_con, 3)))
