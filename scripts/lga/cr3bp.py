import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class CR3BP:

	def __init__(self, mu, L, V, T):
		""" A CR3BP problem is only defined by its parameter mu """

		# Characteristic parameter
		self.mu = mu # [-]

		# Dimensional values
		self.L = L # [km]
		self.V = V # [km/s]
		self.T = T # [s]

	def states_derivatives(self, t, r):
		""" Computation of the derivatives of a vector [x, y, z, vx, vy, vz]^T """

		x, y, z, vx, vy, vz = r

		U_bar_x, U_bar_y, U_bar_z = self.pseudo_potential_derivatives(r)

		x_dot = vx
		y_dot = vy
		z_dot = vz

		vx_dot = 2*vy - U_bar_x
		vy_dot = -2*vx - U_bar_y
		vz_dot = - U_bar_z

		return np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])

	def pseudo_potential(self, r):
		""" Computation of the pseudo potential U_bar """

		x, y, z = r[0:3]

		r_1 = (x + self.mu) ** 2 + y ** 2 + z ** 2
		r_2 = (x - 1 + self.mu) ** 2 + y ** 2 + z ** 2

		U_bar = - 0.5 * ((1 - self.mu) * r_1**2 + self.mu * r_2**2) - (1 - self.mu)/(r_1**0.5) - self.mu/(r_2**0.5)

		return U_bar

	def pseudo_potential_derivatives(self, r):
		""" Computation of the derivatives of the pseudo-potential U_bar wrt. x, y and z """

		x, y, z = r[0:3]

		r_1 = (x + self.mu) ** 2 + y ** 2 + z ** 2
		r_2 = (x - 1 + self.mu) ** 2 + y ** 2 + z ** 2

		U_bar_x = self.mu * (x - 1 + self.mu) / r_2 ** (3 / 2) + \
			(1 - self.mu) * (x + self.mu) / r_1 ** (3 / 2) - x

		U_bar_y = self.mu * y / r_2 ** (3 / 2) + (1 - self.mu) * y / r_1 ** (3 / 2) \
			- y

		U_bar_z = self.mu * z / \
			r_2 ** (3 / 2) + (1 - self.mu) * z / r_1 ** (3 / 2)

		return U_bar_x, U_bar_y, U_bar_z
 
if __name__ == '__main__':

	cr3bp = CR3BP(mu=1.215e-2)

	y0 = [0.2, 0, 0, 0, 2.3, 0]

	t_span = [0, 10]
	t_eval = np.linspace(t_span[0], t_span[1], 1000)
	sol = solve_ivp(fun=cr3bp.states_derivatives, t_span=t_span, t_eval=t_eval, y0=y0, method='RK45')

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot(sol.y[0, :], sol.y[1, :], sol.y[2, :])
	ax.plot([-cr3bp.mu], [0], [0], 'o', color='black', markersize=9, label='Earth')
	ax.plot([1-cr3bp.mu], [0], [0], 'o', color='black', markersize=5, label='Moon')

	plt.legend()
	plt.show()


