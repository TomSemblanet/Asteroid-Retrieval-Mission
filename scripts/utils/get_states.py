import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl

import matplotlib.pyplot as plt

from scripts.udp.NEA_Earth_UDP import NEA2Earth
from scripts.utils import load_sqp, load_kernels, load_bodies
from scripts.utils.post_process import post_process

from data import constants as cst


def taylor_disturbance(r0, v0, m0, thrust, disturbance, tof, mu, veff, N=60):

    # We define the integration time ...
    dt = tof / (N - 1)

    # ... and calcuate the cartesian components for r
    x = [0.0] * N
    y = [0.0] * N
    z = [0.0] * N

    vx = [0.0] * N
    vy = [0.0] * N
    vz = [0.0] * N

    m_ = [0.0] * N

    # Replace r0, v0 and m0
    r = r0
    v = v0
    m = m0
    # We calculate the spacecraft position at each dt
    for i in range(N):
        x[i] = r[0]
        y[i] = r[1]
        z[i] = r[2]

        vx[i] = v[0]
        vy[i] = v[1]
        vz[i] = v[2]

        m_[i] = m

        r, v, m = pk.propagate_taylor_disturbance(
            r, v, m, thrust, disturbance, dt, mu, veff, -10, -10)

    r_ = np.array([x, y, z])
    v_ = np.array([vx, vy, vz])

    return r_, v_, m_

def get_states(udp, population, N):

	# Loading the main kernels
	load_kernels.load()

	# Extraction of the decision vector
	x = population.get_x()[0]

	# Extraction of the number of segment of forward and backward propagation
	n_seg = udp.n_seg 
	n_fwd_seg = udp.n_fwd_seg 
	n_bwd_seg = udp.n_bwd_seg

	# Array containing the positon, velocity and thrusts
	R = np.ndarray(shape=(3, N * (n_fwd_seg + n_bwd_seg)))

	# Time of flight
	t0 = x[0]
	tof = x[1]

	# Spacecraft characteristic
	isp = udp.sc.isp
	veff = isp * pk.G0

	# Reconstruction of the forward and backward grid
	fwd_grid = t0 + tof * udp.fwd_grid
	bwd_grid = t0 + tof * udp.bwd_grid

	# Thrust
	throttles = [x[5 + 3 * i : 8 + 3 * i] for i in range(n_seg)]

	# Time vector
	times = np.concatenate((fwd_grid, bwd_grid))

	# Propagation 
	rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, ufwd, ubwd, fwd_dt, bwd_dt, dfwd, dbwd = udp.propagate(x)

	plt.plot([x[0] / pk.AU for x in dbwd], 'x')
	plt.show()

	# ============     FORWARD     ===========================
	for i in range(udp.n_fwd_seg):

		r3 = sum([r**2 for r in dfwd[i]])**(3 / 2)
		disturbance = [mfwd[i] * pk.MU_EARTH / r3 * ri for ri in dfwd[i]]

		r_fwd__, v_fwd__, m_fwd__ = taylor_disturbance(rfwd[i], vfwd[i], mfwd[i], ufwd[i], disturbance, fwd_dt[i], \
			cst.MU_SUN, veff, N=N)

		# Adding the position vectors
		if i < (udp.n_fwd_seg - 1):
			R[:, i*(N-1): (i+1)*(N-1)] = r_fwd__[:, :-1]
		else:
			R[:, i*(N-1): (i+1)*(N-1)+1] = r_fwd__

	# =======================================================

	# plt.plot(R[0, :((udp.n_fwd_seg - 1)+1)*(N-1)+1], label='x')
	# plt.plot(R[1, :((udp.n_fwd_seg - 1)+1)*(N-1)+1], label='y')
	# plt.plot(R[2, :((udp.n_fwd_seg - 1)+1)*(N-1)+1], label='z')

	# plt.grid()
	# plt.legend()

	# plt.show()

	for i in range(n_bwd_seg):
		r3 = sum([r**2 for r in dbwd[-1 - i]])**(3 / 2)
		disturbance = [mbwd[i] * pk.MU_EARTH / r3 * ri for ri in dbwd[-1 - i]]

		r_bwd__, v_bwd__, m_bwd__ = taylor_disturbance(rbwd[-1 - i], vbwd[-1 - i], mbwd[-1 - i], ubwd[-1 - i], disturbance, -bwd_dt[-1 - i], \
			cst.MU_SUN, veff, N=N)

		# Adding the position vectors
		if i == 0:
			R[:, -1 - ((i+1)*(N-1)) : -1 - (i*(N-1))] = np.flip(r_bwd__[:, 1:-1], axis=1)

	plt.plot(R[0], label='x')
	plt.plot(R[1], label='y')
	plt.plot(R[2], label='z')

	plt.grid()
	plt.legend()

	plt.show()


if __name__ == '__main__':

	with open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/08_04_2021_results/2044', 'rb') as f:
		res = pkl.load(f)

	# load_kernels.load()
	# post_process(res['udp'], res['population'].get_x()[0])

	get_states(res['udp'], res['population'], 20)