
import matplotlib.pyplot as plt
import numpy as np
import unittest

from src import constraints as cs, cost as ct, collocation as cl, pseudospectrals as ps, transcription, problem, utils


class ProblemTest(unittest.TestCase):

    def test_trapez_compute_defects(self):
        """ Tests the Trapezoidal class `compute_defects` method """

        # Number of nodes and states
        self.n_nodes = 3
        self.n_states = 2

        # Instanciation of a minimal Trapezoidal object in order
        # to test the `compute_defect` method
        options = {'h': np.array(
            [1, 2]), 'n_nodes': self.n_nodes, 'n_states': self.n_states}
        trapezoidal = cl.Trapezoidal(options)

        # Computes the derivatives a state vector
        def dynamics(states, controls, f_par):
            dynamics = np.ndarray(
                shape=(self.n_states, self.n_nodes), dtype=float)

            x1 = states[0]
            x2 = states[1]

            x1_dot = x2
            x2_dot = x1 * x2

            dynamics[0] = x1_dot
            dynamics[1] = x2_dot

            return dynamics

        # States vector
        states = np.array([[1, 2, 3], [4, 5, 6]])

        # Controls vector
        controls = np.empty(0)

        # Free parameters
        f_prm = np.empty(0)

        # Computation of defects
        computed_def = trapezoidal.compute_defects(
            states, controls, f_prm, dynamics, 1)

        # Correct defects hand-computed (`h` isn't constant !)
        correct_def = np.array([[-3.5, -10], [-6, -27]])

        # Equality verification
        self.assertTrue(np.array_equal(computed_def, correct_def))

    def test_trapez_quadrature(self):
        """ Tests the Trapezoidal class `quadrature` method """

        # Number of nodes
        N = 10000

        # x-abscissa values
        x = np.linspace(0, 10, N)

        # abscissa steps
        h = x[1:] - x[:-1]

        # Instanciation of a minimal Trapezoidal object in order
        # to test the `compute_defect` method
        options = {'h': h, 'n_nodes': N, 'n_states': 1}
        trapezoidal = cl.Trapezoidal(options)

        # Computation of the values of the exponential
        # function
        func_values = np.exp(x)

        # Computation of the integral
        computed_integral = trapezoidal.quadrature(func_values)

        # Correct value of the integral
        correct_integral = np.exp(10) - np.exp(0)

        # Equality verification (we round the results as the
        # quadrature will never match the exact integral value)
        self.assertEqual(round(computed_integral, 2),
                         round(correct_integral, 2))

    def test_trapez_nodes_adaptation(self):
        """ Tests the Trapezoidal class `nodes_adaptation` method """

        # Number of nodes
        N = 10

        # Unscaled time
        uscl_time = np.array([0, 2, 5, 6, 7, 12, 13, 20, 22, 25])

        # Unscaled steps
        uscl_h = uscl_time[1:] - uscl_time[:-1]

        # Instanciation of a minimal Trapezoidal object in order
        # to test the `compute_defect` method
        options = {'h': uscl_h, 'n_nodes': N, 'n_states': 1}
        trapezoidal = cl.Trapezoidal(options)

        # Computation of empty states and controls matrices
        # needed in real situation by the method
        x = u = []

        # Computation of time-scaling factor
        t_i, t_f = uscl_time[0], uscl_time[-1]
        scl_fact = (t_f - t_i) / 2

        # Computation of the new time-steps
        x_, u_, computed_h = trapezoidal.nodes_adaptation(x, u, uscl_time)

        # Correct scaled time_steps
        correct_h = uscl_h / scl_fact

        # Equality verification
        self.assertTrue(np.array_equal(correct_h, np.round(computed_h, 2)))

    def test_hermite_simpson_compute_states_col(self):
        """ Test the Hermite-Simpson class `compute_states_col` method """

        # Instanciation of a minimal HermiteSimpson object in order
        # to test the `compute_col_points` method
        options = {'h': np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1]), 'n_nodes': 9, 'n_states': 1}
        hermite_simpson = cl.HermiteSimpson(options)
        sc_factor = 1

        # Definition of test states and controls
        x = np.array([np.cos(range(10))])
        u = np.array([np.sin(range(10))])
        f_par = np.empty(0)

        # Definition of dynamics
        def dynamics(states, controls, f_par):
            return np.array([-np.sin(range(10))])

        # Computation of states and controls at col-points
        cpt_st_col = hermite_simpson.compute_states_col(
            x, u, f_par, dynamics, sc_factor)
        cor_st_col = np.array(
            [[0.875,  0.070, -0.799, -0.934, -0.210,  0.706, 0.974,  0.345, -0.600]])

        # Equality check
        self.assertTrue(np.array_equal(
            np.round(cpt_st_col[0], 1), np.round(cor_st_col[0], 1)))

    def test_hermite_simpson_compute_defects(self):
        """ Tests the Hermite-Simpson class `compute_defects` method """

        # Instanciation of a minimal HermiteSimpson object in order
        # to test the `compute_defect` method
        options = {'h': np.array([1, 2]), 'n_states': 2,
                   'n_controls': 1, 'n_nodes': 3}
        hermite_simpson = cl.HermiteSimpson(options)
        sc_factor = 1

        # Computes the derivatives a state vector
        def dynamics(states, controls, f_par):
            dynamics = np.ndarray(
                shape=(states.shape[0], states.shape[1]), dtype=float)

            x1 = states[0]
            x2 = states[1]

            u = controls[0]

            x1_dot = x2 + u
            x2_dot = x1 * x2

            dynamics[0] = x1_dot
            dynamics[1] = x2_dot

            return dynamics

        # Number of nodes and states
        n_nodes = 3
        n_states = 2
        n_controls = 1

        # States vector
        states = np.array([[1, 2, 3], [4, 5, 6]])

        # Controls vector
        controls = np.array([[1, 1, 1]])
        controls_col = np.array([[1, 1]])

        # Free parameters
        f_par = np.empty(0)

        # Computation of col-points
        states_col = hermite_simpson.compute_states_col(
            states, controls, f_par, dynamics, sc_factor)

        # Computation of defects
        computed_def = hermite_simpson.compute_defects(
            states, controls, controls_col, f_par, dynamics, sc_factor)

        # Correct defects hand-computed (`h` isn't constant !)
        correct_def = np.array([[-4., -9.33], [-4.77, -18.83]])

        # Equality verification
        self.assertTrue(np.array_equal(np.round(computed_def, 2), correct_def))

    def test_hermite_simpson_quadrature(self):
        """ Tests the Hermite-Simpson class `quadrature` method """

        # Number of nodes
        N = 10000

        # x-abscissa values
        x = np.linspace(0, 10, N)

        # abscissa steps
        h = x[1:] - x[:-1]

        # x-abcissa col-values
        x_col = [x[k] + h[k]/2 for k in range(N-1)]

        def exp_(x, *args):
            return np.exp(x)

        # Instanciation of a minimal Trapezoidal object in order
        # to test the `compute_defect` method
        options = {'h': h, 'n_nodes': N, 'n_states': 1}
        hermite_simpson = cl.HermiteSimpson(options)

        # Computation of the values of the exponential
        # function
        func_values = np.exp(x)

        # Computation of the values of the exponential
        # function at col-points
        func_valus_col = np.exp(x_col)

        # Computation of the integral
        computed_integral = hermite_simpson.quadrature(
            func_values, func_valus_col)

        # Correct value of the integral
        correct_integral = np.exp(10) - np.exp(0)

        # Equality verification (we round the results as the
        # quadrature will never match the exact integral value)
        self.assertEqual(round(computed_integral, 2),
                         round(correct_integral, 2))

    def test_hermite_simpson_nodes_adaptation(self):
        """ Tests the HermiteSimpson class `nodes_adaptation` method """

        # Number of nodes
        N = 10

        # Unscaled time
        uscl_time = np.array([0, 2, 5, 6, 7, 12, 13, 20, 22, 25])

        # Unscaled steps
        uscl_h = uscl_time[1:] - uscl_time[:-1]

        # Instanciation of a minimal Trapezoidal object in order
        # to test the `compute_defect` method
        options = {'h': uscl_h, 'n_nodes': N, 'n_states': 1}
        hermite_simpson = cl.HermiteSimpson(options)

        # Computation of empty states and controls matrices
        # needed in real situation by the method
        x = u = []

        # Computation of time-scaling factor
        t_i, t_f = uscl_time[0], uscl_time[-1]
        scl_fact = (t_f - t_i) / 2

        # Computation of the new time-steps
        x_, u_, computed_h = hermite_simpson.nodes_adaptation(x, u, uscl_time)

        # Correct scaled time_steps
        correct_h = uscl_h / scl_fact

        # Equality verification
        self.assertTrue(np.array_equal(correct_h, np.round(computed_h, 2)))

    def test_hermite_simpson_interpolate_ctrl(self):
        """ Tests the HermiteSimpson class `interpolate_ctrl` method """

        # Number of nodes
        N = 10000

        # Unscaled time
        time = np.linspace(0, 10, N)

        # Unscaled steps
        h = time[1:] - time[:-1]

        # Mid-points
        time_col = np.array([t_ + h_/2 for t_, h_ in zip(time, h)])

        # Instanciation of a minimal Hermite-Simpson object in order
        # to test the `interpolate_ctrl` method
        options = {'h': h, 'n_nodes': N, 'n_states': 1, 'n_controls': 1}
        hermite_simpson = cl.HermiteSimpson(options)

        controls = np.array([np.cos(time)])
        controls_col = np.array([np.cos(time_col)])

        hermite_simpson.build_interpolation_func(time, controls, controls_col)

        intrp_controls = []
        for t_intrp in np.linspace(0, 10, N-1):
            control_intrp = hermite_simpson.interpolate_ctrl(
                time, controls, t_intrp)
            intrp_controls.append(control_intrp[0])

        cpt_intrp_controls = np.array(intrp_controls)
        cor_intrp_controls = np.array(np.cos(np.linspace(0, 10, N-1)))

        # Equality check
        self.assertTrue(np.array_equal(
            np.round(cpt_intrp_controls, 11), np.round(cor_intrp_controls, 11)))
