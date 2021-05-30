import sys
import pickle 
import numpy as np 
import matplotlib.pyplot as plt

import cppad_py

from scipy.interpolate import interp1d

from collocation.GL_V.src.problem import Problem
from collocation.GL_V.src.optimization import Optimization

class ApogeeRaising(Problem):
	""" CR3BP : Moon-Moon Leg optimal control problem """

	def __init__(self, cr3bp, mass0, Tmax, trajectory, time, v_inf):
		""" Initialization of the `GoddardRocket` class """
		n_states = 7
		n_controls = 4
		n_st_path_con = 0
		n_ct_path_con = 1
		n_event_con = 9
		n_f_par = 0
		n_nodes = 400

		Problem.__init__(self, n_states, n_controls, n_st_path_con, n_ct_path_con, 
						 n_event_con, n_f_par, n_nodes)

		# Set some attributs
		self.cr3bp = cr3bp 

		self.mass0 = mass0 # [kg]
		self.Tmax = Tmax   # [kN]

		self.trajectory = trajectory # [L] | [L/T]
		self.time = time # [T]

		self.v_inf = v_inf # [km/s]

	def set_constants(self):
		""" Setting of the problem constants """
		self.Tmax /= self.cr3bp.L / self.cr3bp.T**2   # Thrusts dimensioning

		self.v_inf /= self.cr3bp.V 	# Excess velocity dimensioning

		self.g0 = 9.80665e-3 / (self.cr3bp.L / self.cr3bp.T**2)
		self.Isp = 3000 / self.cr3bp.T

	def set_boundaries(self):
		""" Setting of the states, controls, free-parameters, initial and final times
						boundaries """

		# States boundaries
		# X [-]
		self.low_bnd.states[0] = -2
		self.upp_bnd.states[0] =  2

		# Y [-]
		self.low_bnd.states[1] = -2
		self.upp_bnd.states[1] =  2

		# Z [-]
		self.low_bnd.states[2] = -2
		self.upp_bnd.states[2] =  2

		# Vx [-]
		self.low_bnd.states[3] = -15
		self.upp_bnd.states[3] =  15

		# Vy [-]
		self.low_bnd.states[4] = -15
		self.upp_bnd.states[4] =  15

		# Vz [-]
		self.low_bnd.states[5] = -15
		self.upp_bnd.states[5] =  15

		# m [kg]
		self.low_bnd.states[6] = 1e-6
		self.upp_bnd.states[6] = self.mass0


		# T [-]
		self.low_bnd.controls[0] = 1e-6
		self.upp_bnd.controls[0] = self.Tmax

  		# Tx [-]
		self.low_bnd.controls[1] = - 1
		self.upp_bnd.controls[1] =   1

		# Ty [-]
		self.low_bnd.controls[2] = - 1
		self.upp_bnd.controls[2] =   1

		# Tz [-]
		self.low_bnd.controls[3] = - 1
		self.upp_bnd.controls[3] =   1


		# Initial and final times boundaries
		self.low_bnd.ti = self.upp_bnd.ti = self.time[0]
		self.low_bnd.tf = 0.9 * self.time[-1]
		self.upp_bnd.tf = 1.1 * self.time[-1]


	def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
		""" Computation of the events constraints """
		events = np.ndarray((self.prm['n_event_con'], 1),
							dtype=cppad_py.a_double)

		x_i, y_i, z_i, vx_i, vy_i, vz_i, m_i = xi
		x_f, y_f, z_f, vx_f, vy_f, vz_f, m_f = xf

		d_f = ((x_f - (1 - self.cr3bp.mu))**2 + y_f**2 + z_f**2) ** (0.5)
		v_f = (vx_f**2 + vy_f**2 + vz_f**2 ) ** (0.5)

		events[0] = x_i  - self.trajectory[0, 0]
		events[1] = y_i  - self.trajectory[1, 0]
		events[2] = z_i  - self.trajectory[2, 0]
		events[3] = vx_i - self.trajectory[3, 0]
		events[4] = vy_i - self.trajectory[4, 0]
		events[5] = vz_i - self.trajectory[5, 0]
		events[6] = m_i - self.mass0

		events[7] = d_f - 10000 / self.cr3bp.L
		events[8] = v_f - self.v_inf

		return events


	def set_events_constraints_boundaries(self):
		""" Setting of the events constraints boundaries """
		self.low_bnd.event[0] = self.upp_bnd.event[0] = 0
		self.low_bnd.event[1] = self.upp_bnd.event[1] = 0
		self.low_bnd.event[2] = self.upp_bnd.event[2] = 0
		self.low_bnd.event[3] = self.upp_bnd.event[3] = 0
		self.low_bnd.event[4] = self.upp_bnd.event[4] = 0
		self.low_bnd.event[5] = self.upp_bnd.event[5] = 0
		self.low_bnd.event[6] = self.upp_bnd.event[6] = 0

		self.low_bnd.event[7] = self.upp_bnd.event[7] = 0
		self.low_bnd.event[8] = self.upp_bnd.event[8] = 0

	def path_constraints(self, states, controls, states_add, controls_add, controls_col, f_par):

		st_path = np.ndarray((self.prm['n_st_path_con'],
							2*self.prm['n_nodes']-1), dtype=cppad_py.a_double)
		ct_path = np.ndarray((self.prm['n_ct_path_con'],
							4*self.prm['n_nodes']-3), dtype=cppad_py.a_double)

		# Thrust magnitude in x, y and z directions in the synodic frame [-]
		ux = np.concatenate((controls[1], controls_add[1], controls_col[1]))
		uy = np.concatenate((controls[2], controls_add[2], controls_col[2]))
		uz = np.concatenate((controls[3], controls_add[3], controls_col[3]))

		u2 = ux*ux + uy*uy + uz*uz

		ct_path[0] = u2 - 1

		return st_path, ct_path

	def set_path_constraints_boundaries(self):
		""" Setting of the path constraints boundaries """
		self.low_bnd.ct_path[0] = self.upp_bnd.ct_path[0] = 0


	def dynamics(self, states, controls, f_prm, expl_int=False):
		""" Computation of the states derivatives """
		if expl_int == False:
			dynamics = np.ndarray(
				(states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
		else:
			dynamics = np.zeros(len(states))

		# Mass [kg]
		m = states[6]

		# Extraction of controls
		T = controls[0]
		ux, uy, uz = controls[1:]

		x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot = self.cr3bp.states_derivatives(0, states[:-1])

		dynamics[0] = x_dot
		dynamics[1] = y_dot
		dynamics[2] = z_dot

		dynamics[3] = vx_dot + T / m * ux
		dynamics[4] = vy_dot + T / m * uy
		dynamics[5] = vz_dot + T / m * uz

		dynamics[6] = - T / self.Isp / self.g0

		return dynamics

	def end_point_cost(self, ti, xi, tf, xf, f_prm):
		""" Computation of the end point cost (Mayer term) """
		m_f = xf[-1]

		return - m_f / self.mass0


	def set_initial_guess(self):
		""" Setting of the initial guess for the states, controls, free-parameters
						and time grid """

		# Interpolation of the states
		f_x = interp1d(self.time, self.trajectory[0])
		f_y = interp1d(self.time, self.trajectory[1])
		f_z = interp1d(self.time, self.trajectory[2])

		f_vx = interp1d(self.time, self.trajectory[3])
		f_vy = interp1d(self.time, self.trajectory[4])
		f_vz = interp1d(self.time, self.trajectory[5])

		# Time
		self.initial_guess.time = np.linspace(self.time[0], self.time[-1], self.prm['n_nodes'])

		# States
		self.initial_guess.states = np.ndarray(
			shape=(self.prm['n_states'], self.prm['n_nodes']))

		self.initial_guess.states[0] = f_x(self.initial_guess.time)
		self.initial_guess.states[1] = f_y(self.initial_guess.time)
		self.initial_guess.states[2] = f_z(self.initial_guess.time)

		self.initial_guess.states[3] = f_vx(self.initial_guess.time)
		self.initial_guess.states[4] = f_vy(self.initial_guess.time)
		self.initial_guess.states[5] = f_vz(self.initial_guess.time)

		self.initial_guess.states[6] = self.mass0 * np.ones(self.prm['n_nodes'])

		# Controls
		self.initial_guess.controls = np.ndarray(
			shape=(self.prm['n_controls'], self.prm['n_nodes']))

		self.initial_guess.controls = np.zeros((4, self.prm['n_nodes']))


if __name__ == '__main__':

	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/orbit_raising_cr3bp/29-05-2021-separated', 'rb') as file:
		results = pickle.load(file)

	v_out = np.array([ 0.84168181508,   0.09065171796, -0.27474864627])

	apogee_raising_problem = ApogeeRaising(results['cr3bp'], 2000, 2/1000, results['syn_ut_traj'], results['syn_ut_time'], np.linalg.norm(v_out))

	# Instantiation of the optimization
	optimization = Optimization(problem=apogee_raising_problem)

	# Launch of the optimization
	optimization.run()

	opt_trajectory = optimization.results['opt_st']


	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(opt_trajectory[0], opt_trajectory[1], '-')
	ax.plot([-results['cr3bp'].mu], [0], 'o', color='black', markersize=5)
	ax.plot([1-results['cr3bp'].mu], [0], 'o', color='black', markersize=2)

	plt.grid()
	plt.show()