import sys
import pykep as pk 
import pygmo as pg 
import numpy as np
import pickle as pkl

import matplotlib.pyplot as plt

from scripts.udp.Earth_NEA.Earth_NEA_UDP import Earth2NEA
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
	tf = x[0]
	tof = x[1]

	# Spacecraft characteristic
	isp = udp.sc.isp
	veff = isp * pk.G0

	# Reconstruction of the forward and backward grid
	fwd_grid = (tf - tof) + tof * udp.fwd_grid
	bwd_grid = (tf - tof) + tof * udp.bwd_grid

	# Thrust
	throttles = [x[6 + 3 * i : 9 + 3 * i] for i in range(n_seg)]

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

	# Correct the mass error 
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
		fig = plt.figure(figsize=(7, 7))
		ax = fig.gca(projection='3d')

		pk.orbit_plots.plot_planet(plnt=udp.earth, t0=pk.epoch(T[0]), tf=pk.epoch(T[-1]), N=1000, axes=ax, s=0, color='orange', legend=(False, "Earth orbit"))
		pk.orbit_plots.plot_planet(plnt=udp.nea, t0=pk.epoch(T[0]), tf=pk.epoch(T[-1]), N=1000, axes=ax, s=0, color='green', legend=(False, "CD3-2020 orbit"))
		ax.plot([0], [0], [0], marker='o', markersize=5, color='yellow')

		ax.plot(R[0], R[1], R[2], label="Spacecraft trajectory")

		ax.plot([R[0,0]], [R[1, 0]], [R[2, 0]], 'o', markersize=3, color='black', label="Departure position")
		ax.plot([R[0, -1]], [R[1, -1]], [R[2, -1]], 'o', markersize=3, color='red', label="Arrival position")

		ax.set_xlim(-2e11, 2e11)
		ax.set_ylim(-2e11, 2e11)
		ax.set_zlim(-1e10, 1e10)

		ax.set_xlabel("x [m]")
		ax.set_ylabel("y [m]")
		ax.set_zlabel("z [m]")

		plt.title("Earth to CD3-2020 trajectory")

		plt.legend()
		plt.show()

	return R, V, M, U, T

if __name__ == '__main__':

	# Pickle file to load
	file = sys.argv[1]

	with open(file, 'rb') as f:
		res = pkl.load(f)

	# Loads the Spice kernels
	load_kernels.load()

	R, V, M, U, T = get_states(udp=res['udp'], population=res['population'], N=50, plot=True)

	# # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * 
	# # Plot of the Spacecraft departure from the Earth-Moon system
	# # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * 

	# earth = load_bodies.planet('EARTH')
	# moon = load_bodies.planet('MOON', observer='EARTH')

	# # Convert the spacecraft coordinates into the Geocentric frame
	# for i, T_ in enumerate(T):
	# 	earth_r = earth.eph(T_)[0]
	# 	earth_v = earth.eph(T_)[1]

	# 	R[0, i] -= earth_r[0]
	# 	R[1, i] -= earth_r[1]
	# 	R[2, i] -= earth_r[2]

	# 	V[0, i] -= earth_v[0]
	# 	V[1, i] -= earth_v[1]
	# 	V[2, i] -= earth_v[2]


	# fig = plt.figure()
	# ax = fig.gca(projection='3d')

	# pk.orbit_plots.plot_planet(plnt=moon, t0=pk.epoch(T[0]), tf=pk.epoch(T[1000]), N=1000, axes=ax, s=0, color='green', legend=(False, "Moon orbit"))

	# ax.plot(R[0, :1000], R[1, :1000], R[2, :1000], label='Spacecraft trajectory')
	# ax.plot(moon.eph(T[0])[0][0], moon.eph(T[0])[0][1], moon.eph(T[0])[0][2], 'o', label='Moon at departure')
	# ax.plot([0], [0], [0], 'o', markersize=10, label='Earth')

	# alpha=5e5

	# moon_r, moon_v = moon.eph(T[0])
	# ax.plot([moon_r[0], moon_r[0]+V[0, 0]*alpha], [moon_r[1], moon_r[1]+V[1, 0]*alpha], \
	# 	[moon_r[2], moon_r[2]+V[2, 0]*alpha], label='Spacecraft velocity')

	# ax.plot([moon_r[0], moon_r[0]+moon_v[0]*alpha], [moon_r[1], moon_r[1]+moon_v[1]*alpha], \
	# 	[moon_r[2], moon_r[2]+moon_v[2]*alpha], label='Spacecraft velocity')
	# # ax_.plot([0, moon_helio.eph(T[0])[1][0]], [0, moon_helio.eph(T[0])[1][1]], [0, moon_helio.eph(T[0])[1][2]], label='Moon velocity')


	# ax.set_xlim(-1e9, 1e9)
	# ax.set_ylim(-1e9, 1e9)
	# ax.set_zlim(-1e9, 1e9)

	# plt.legend()
	# plt.show()

	