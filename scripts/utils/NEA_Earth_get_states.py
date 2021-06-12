import sys
import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl

import matplotlib.pyplot as plt

from scripts.udp.NEA_Earth.NEA_Earth_UDP import NEA2Earth
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

def get_states(udp, population, N, plot):

	# Extraction of the decision vector
	x = population.get_x()[0]

	# Extraction of the number of segment on forward and backward propagation
	n_seg = udp.n_seg 
	n_fwd_seg = udp.n_fwd_seg
	n_bwd_seg = udp.n_bwd_seg

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
	times = np.concatenate((fwd_grid, bwd_grid[1:]))
	dts = times[1:] - times[:-1]

	# Propagation 
	rfwd, rbwd, vfwd, vbwd, mfwd, mbwd, ufwd, ubwd, fwd_dt, bwd_dt, dfwd, dbwd = udp.propagate(x)

	# ============     FORWARD     =================

	buffer_rfwd = list() # Position (r)
	buffer_vfwd = list() # Velocity (v)
	buffer_mfwd = list() # Mass (m)
	buffer_ufwd = list() # Throttles (u)
	buffer_tfwd = list() # Time (t)

	for i in range(n_fwd_seg):

		r3 = sum([r**2 for r in dfwd[i]])**(3 / 2)
		disturbance = [mfwd[i] * pk.MU_EARTH / r3 * ri for ri in dfwd[i]]

		r_fwd__, v_fwd__, m_fwd__ = taylor_disturbance(rfwd[i], vfwd[i], mfwd[i], ufwd[i], disturbance, fwd_dt[i], \
			cst.MU_SUN, veff, N=N)

		if i == 0:
			buffer_rfwd = r_fwd__[:, :-1]
			buffer_vfwd = v_fwd__[:, :-1]
			buffer_mfwd = m_fwd__[:-1]
			buffer_ufwd = np.transpose(np.repeat([ufwd[i]], (N), axis=0))
			buffer_tfwd = np.array([times[i] + (j * fwd_dt[i] / (N-1)) * cst.SEC2DAY for j in range(N-1)])

		elif (i > 0 and i < (n_fwd_seg - 1)):
			buffer_rfwd = np.append(buffer_rfwd, r_fwd__[:, :-1], axis=1)
			buffer_vfwd = np.append(buffer_vfwd, v_fwd__[:, :-1], axis=1)
			buffer_mfwd = np.append(buffer_mfwd, m_fwd__[:-1], axis=0)
			buffer_ufwd = np.append(buffer_ufwd, np.transpose(np.repeat([ufwd[i]], (N-1), axis=0)), axis=1)
			buffer_tfwd = np.append(buffer_tfwd, [times[i] + (j * fwd_dt[i] / (N-1)) * cst.SEC2DAY for j in range(N-1)])

		else:
			buffer_rfwd = np.append(buffer_rfwd, r_fwd__, axis=1)
			buffer_vfwd = np.append(buffer_vfwd, v_fwd__, axis=1)
			buffer_mfwd = np.append(buffer_mfwd, m_fwd__, axis=0)
			buffer_ufwd = np.append(buffer_ufwd, np.transpose(np.repeat([ufwd[i]], (N-1), axis=0)), axis=1)
			buffer_tfwd = np.append(buffer_tfwd, [times[i] + (j * fwd_dt[i] / N) * cst.SEC2DAY for j in range(N)])

	# ============     BACKWARD     =================

	buffer_rbwd = list()
	buffer_vbwd = list()
	buffer_mbwd = list()
	buffer_ubwd = list()
	buffer_tbwd = list()

	for i in range(n_bwd_seg):
		r3 = sum([r**2 for r in dbwd[-1 - i]])**(3 / 2)
		disturbance = [mbwd[-1 - i] * pk.MU_EARTH / r3 * ri for ri in dbwd[-1 - i]]

		r_bwd__, v_bwd__, m_bwd__ = taylor_disturbance(rbwd[-1 - i], vbwd[-1 - i], mbwd[-1 - i], ubwd[-1 - i], disturbance, -bwd_dt[-1 - i], \
			cst.MU_SUN, veff, N=N)

		if i == 0:
			buffer_rbwd = r_bwd__
			buffer_vbwd = v_bwd__
			buffer_mbwd = m_bwd__
			buffer_ubwd = np.transpose(np.repeat([ubwd[i]], N, axis=0))
			buffer_tbwd = np.array([times[-1 - i] - (j * bwd_dt[-1 - i] / (N)) * cst.SEC2DAY for j in range(N)])

		else:
			buffer_rbwd = np.append(buffer_rbwd, r_bwd__[:, 1:], axis=1)
			buffer_vbwd = np.append(buffer_vbwd, v_bwd__[:, 1:], axis=1)
			buffer_mbwd = np.append(buffer_mbwd, m_bwd__[1:], axis=0)
			buffer_ubwd = np.append(buffer_ubwd, np.transpose(np.repeat([ubwd[i]], (N-1), axis=0)), axis=1)
			buffer_tbwd = np.append(buffer_tbwd, [times[-1 - i] - (j * bwd_dt[-1 - i] / (N-1)) * cst.SEC2DAY for j in range(N-1)])

	# Flip the matrix because it's backward propagation (not for u)
	buffer_rbwd = np.flip(buffer_rbwd, axis=1)
	buffer_vbwd = np.flip(buffer_vbwd, axis=1)
	buffer_mbwd = np.flip(buffer_mbwd, axis=0)
	buffer_tbwd = np.flip(buffer_tbwd, axis=0)

	# Mass error correction 
	buffer_mbwd += mfwd[-1] - mbwd[0]

	# Concatenation of the forward and backward arrays
	R = np.concatenate((buffer_rfwd, buffer_rbwd), axis=1)
	V = np.concatenate((buffer_vfwd, buffer_vbwd), axis=1)
	M = np.concatenate((buffer_mfwd, buffer_mbwd), axis=0)
	U = np.concatenate((buffer_ufwd, buffer_ubwd), axis=1)
	T = np.concatenate((buffer_tfwd, buffer_tbwd), axis=0)

	# Summary of the transfer
	udp.brief(x)

	if plot == True:
		plt.plot(T, R[0], label='x')
		plt.plot(T, R[1], label='y')
		plt.plot(T, R[2], label='z')

		plt.title("Position")
		plt.legend()
		plt.grid()
		plt.show()


		plt.plot(T, V[0], label='vx')
		plt.plot(T, V[1], label='vy')
		plt.plot(T, V[2], label='vz')

		plt.title("Velocity")
		plt.legend()
		plt.grid()
		plt.show()


		plt.plot(T, M, label='m')

		plt.title("Mass")
		plt.legend()
		plt.grid()
		plt.show()

		plt.plot(T, U[0], label='ux')
		plt.plot(T, U[1], label='uy')
		plt.plot(T, U[2], label='uz')

		plt.title("Control")
		plt.legend()
		plt.grid()
		plt.show()

		# 3-D Plot
		fig = plt.figure()
		ax = fig.gca(projection='3d')

		pk.orbit_plots.plot_planet(plnt=udp.earth, t0=T[0], tf=T[0]+366, N=1000, axes=ax, s=0, color='red', alpha=0.7, legend=(False, "Earth orbit"))
		pk.orbit_plots.plot_planet(plnt=udp.nea, t0=T[0], tf=T[-1], N=1000, axes=ax, s=0, color='green', alpha=0.7, legend=(False, "CD3-2020 orbit"))
		ax.plot(R[0], R[1], R[2], color='blue', alpha=0.7, linewidth=1, label="Spacecraft trajectory")
		
		ax.plot([0], [0], [0], 'o', markersize=10, color='yellow', label="Sun")
		ax.plot([R[0,0]], [R[1, 0]], [R[2, 0]], 'o', markersize=3, color='black', label="Departure position")
		ax.plot([R[0, -1]], [R[1, -1]], [R[2, -1]], 'o', markersize=3, color='red', label="Arrival position")

		ax.set_xlim(-2e11, 2e11)
		ax.set_ylim(-2e11, 2e11)
		ax.set_zlim(-7e9, 7e9)

		ax.set_xlabel("X [m]")
		ax.set_ylabel("Y [m]")
		ax.set_zlabel("Z [m]")

		plt.title("CD3-2020 to Earth trajectory")

		plt.legend()
		plt.show()

	post_process(udp, x)

	return R, V, M, U, T


if __name__ == '__main__':

	# Pickle file to load
	file = sys.argv[1]

	with open(file, 'rb') as f:
		res = pkl.load(f)

	# Loads the Spice kernels
	load_kernels.load()

	# post_process(res['udp'], res['population'].get_x()[0])

	R, V, M, U, T = get_states(udp=res['udp'], population=res['population'], N=50, plot=True)


	