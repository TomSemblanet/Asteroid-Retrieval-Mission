import numpy as np 
import pickle

import matplotlib.pyplot as plt 

from scipy.integrate import solve_ivp

from scripts.earth_departure.cr3bp import CR3BP


def get_state(DROs, Cjac, theta):
	""" Returns the state on a DRO orbit characterized by the Jacobian constant ``Cjac`` (for the orbit) 
		and a parameter ``theta`` representing a point on the orbit """

	Cjacs = np.sort(np.array(list(DROs.keys()), dtype=float))
	index = np.searchsorted(Cjacs, Cjac, side='left') - 1

	# Extraction of the DRO and state of interests
	DRO = DROs[str(Cjacs[index])]
	state = DRO[int(theta * DRO.shape[0]), :]

	return DRO, state


def plot_DRO_state(ax, DRO, state):

	ax.plot(DRO[:, 0], DRO[:, 1], '-', color='orange', linewidth=1)
	ax.plot([state[0]], [state[1]], 'o', color='magenta', markersize=2)	


def propagate(DRO):

	# 1 - Instantiation of the Earth - Moon CR3BP
	# -------------------------------------------
	mu = 0.012151
	L = 384400
	T = 2360591.424
	cr3bp = CR3BP(mu=mu, L=L, V=L/(T/(2*np.pi)), T=T/(2*np.pi))

	t_span = [0, 5]
	t_eval = np.linspace(t_span[0], t_span[-1], 10000)

	solution = solve_ivp(fun=cr3bp.states_derivatives, t_span=t_span, t_eval=t_eval, y0=DRO[0, :], rtol=1e-10, atol=1e-13)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(solution.y[0], solution.y[1], '-', color='orange', linewidth=1)
	ax.plot([1-cr3bp.mu], [0], 'o', color='black', markersize=2)
	ax.plot([ -cr3bp.mu], [0], 'o', color='black', markersize=5)

	plt.grid()
	plt.show()







