import matplotlib.pyplot as plt
import numpy as np 
import math as mt

from scipy.integrate import solve_ivp

from scripts.earth_departure import constants as cst


def kepler(t, r):
	""" Computation of the states derivatives following Keplerian mechanics """

	x, y, z, vx, vy, vz = r

	# Earth distance [km]
	d = np.linalg.norm(r[:3])

	# Position derivatives [km/s]
	x_dot = vx
	y_dot = vy
	z_dot = vz

	# Velocity derivatives [km/s^2]
	vx_dot = - cst.mu_E / d**3 * x
	vy_dot = - cst.mu_E / d**3 * y
	vz_dot = - cst.mu_E / d**3 * z

	return np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])

def kepler_thrust(t, r, mass, T, eps):
	""" Computation of the states derivatives following Keplerian mechanics.
		The S/C thruster are On when it's on an arc defined by the angles [-eps, eps] """

	x, y, z, vx, vy, vz = r

	# Earth distance [km]
	d = np.linalg.norm(r[:3])

	# S/C velocity [km/s]
	v = np.linalg.norm(r[3:])

	# Angle between the Earth and the S/C
	phi = np.arccos( np.dot( np.array([-1, 0, 0]), r[:3]) / d )

	# Thrust On/Off
	thrust_on = 1 if phi <= eps else 0

	# Position derivatives [km/s]
	x_dot = vx
	y_dot = vy
	z_dot = vz

	# Velocity derivatives [km/s^2]
	vx_dot = - cst.mu_E / d**3 * x + thrust_on * T * vx / v / mass
	vy_dot = - cst.mu_E / d**3 * y + thrust_on * T * vy / v / mass
	vz_dot = - cst.mu_E / d**3 * z + thrust_on * T * vz / v / mass

	return np.array([x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot])

def cart2sph(r):
	""" Converts cartesian coordinates to spherical coordinates """
	rho = np.linalg.norm(r)
	phi = mt.atan2(r[1], r[0])
	theta = mt.acos(r[2]/rho)

	return np.array([rho, phi, theta])


def sph2cart(r):
	""" Converts spherical coordinates to cartesian coordinates """
	return np.array([r[0]*np.cos(r[1])*np.sin(r[2]),
					 r[0]*np.sin(r[1])*np.sin(r[2]),
					 r[0]*np.cos(r[2])])


def R1():
	""" Rotation from the basis with axis following : 
			i : Moon's angular momentum
			j : Moon's position r/ Earth
			k : Moon's velocity r/ Earth

				to the basis with axis following :

			i : Moon's position r/ Earth
			j : Moon's velocity r/ Earth
			k : Moon's angular momentum
	"""

	return np.array([[0, 1, 0],
					 [0, 0, 1],
					 [1, 0, 0]])

def R2(theta):
	""" Rotation from the basis with axis following:
			i : Moon's position r/ Earth
			j : Moon's velocity r/ Earth
			k : Moon's angular momentum

				to the basis with axis following :

			i : Moon's position at t0
			j : completing the right-hand rule
			k : Moon's angular momentum
	"""

	return np.array([ [np.cos(theta), -np.sin(theta), 0],
					  [np.sin(theta),  np.cos(theta), 0],
					  [			0,			  0, 1] ])

def R2_6d(theta):
	""" Rotation from the basis with axis following:
			i : Moon's position r/ Earth
			j : Moon's velocity r/ Earth
			k : Moon's angular momentum

				to the basis with axis following :

			i : Moon's position at t0
			j : completing the right-hand rule
			k : Moon's angular momentum
	"""

	return np.array([ [np.cos(theta), -np.sin(theta), 0,             0,              0, 0],
					  [np.sin(theta),  np.cos(theta), 0,             0,              0, 0],
					  [			   0, 	           0, 1,             0,              0, 0],
					  [            0,              0, 0, np.cos(theta), -np.sin(theta), 0],
					  [            0,              0, 0, np.sin(theta),  np.cos(theta), 0],
					  [			   0, 	           0, 0,             0,              0, 1]])


def P_GEO2RVH(gamma):
	""" Passage matrix from Earth inertial frame to RVH frame """
	return np.array([[np.cos(gamma), -np.sin(gamma), 0],
		 			 [np.sin(gamma),  np.cos(gamma), 0],
		 			 [			0,			  0, 1]])	

def P_RVH2GEO(gamma):
	""" Passage matrix from RVH frame to Earth inertial frame """
	return np.array([[ np.cos(gamma),  np.sin(gamma), 0],
		 			 [-np.sin(gamma),  np.cos(gamma), 0],
		 			 [			   0,			   0, 1]])


def P_RVH2HRV():
	""" Passage matrix from RVH frame to HRV frame """
	return np.array([[0, 1, 0],
					 [0, 0, 1],
					 [1, 0, 0]])

def P_HRV2RVH():
	""" Passage matrix from HRV frame to RVH frame """
	return np.array([[0, 0, 1],
					 [1, 0, 0],
					 [0, 1, 0]])

def P_GEO2HRV(gamma):
	""" Passage matrix from Earth inertial frame to HRV frame """
	return P_GEO2RVH(gamma).dot(P_RVH2HRV())

def P_HRV2GEO(gamma):
	""" Passage matrix from HRV frame to Earth inertial frame """
	return P_HRV2RVH().dot(P_RVH2GEO(gamma))

def kep2cart(a, e, i, W, w, ta, mu) : 

	h = np.sqrt(mu*a*(1-e**2))

	r_orbital_plan = (h*h/mu)*(1/(1+e*np.cos(ta)))*np.array([np.cos(ta), np.sin(ta), 0])
	v_orbital_plan = mu/h*np.array([-np.sin(ta), e+np.cos(ta), 0])


	R1 = np.array([ [np.cos(w), -np.sin(w), 0.],
					[np.sin(w),  np.cos(w), 0.],
					[			 0.,			   0., 1.] ])

	R2 = np.array([ [1.,			   0.,			   0.],
					[0.,	  np.cos(i),	 -np.sin(i)],
					[0.,	  np.sin(i),	  np.cos(i)] ])

	R3 = np.array([ [np.cos(W), -np.sin(W),	 0.],
					[np.sin(W),  np.cos(W),	 0.],
					[			 0.,			   0.,	 1.] ])

	r = R3.dot(R2.dot(R1.dot(r_orbital_plan)))
	v = R3.dot(R2.dot(R1.dot(v_orbital_plan)))
	
	return np.concatenate((r, v))


def cart2kep (r, v, mu) : 

	r_mag = np.linalg.norm(r)
	v_mag = np.linalg.norm(v)
	w_mag = np.dot(r, v)/r_mag

	h = np.cross(r, v)
	h_mag = np.linalg.norm(h)

	i = np.arccos(h[2]/h_mag)

	n = np.cross([0, 0, 1], h)
	n_mag = np.linalg.norm(n)

	if(n_mag != 0) : 
		W = np.arccos(n[0]/n_mag)
		if(n[1] < 0) : 
			W = 2*np.pi - W
	else : 
		W = 0

	e = 1/mu*((v_mag*v_mag - mu/r_mag)*r - r_mag*w_mag*v)
	e_mag = np.linalg.norm(e)

	if(n_mag != 0) : 
		if(e_mag > 1e-5) : 
			w = np.arccos(np.dot(n, e)/(n_mag*e_mag))
			if(e[2] < 0) : 
				w = 2*np.pi - w
		else : 
			w = 0
	else : 
		w = (mt.atan2(e[1], e[0]))%(2*np.pi)
		if(np.sign(np.cross(r,v))[2] < 0) : 
			w = 2*np.pi - w

	if(e_mag > 1e-5) : 
		ta = np.arccos(np.dot(e, r)/(e_mag*r_mag))
		if(w_mag < 0) : 
			ta = 2*np.pi - ta
	else : 
		if(n_mag != 0) :
			ta = np.arccos(np.dot(n, r)/(n_mag*r_mag))
			if(np.cross(n, r)[2] < 0) : 
				ta = 2*np.pi - ta
		else : 
			ta = np.arccos(r[0]/r_mag)
			if(v[0] > 0) : 
				ta = 2*np.pi - ta

	a = h_mag*h_mag/(mu*(1-e_mag*e_mag))


	return (a, e_mag, i, W, w, ta)

def moon_reached(t, r, mass=None, T=None, eps=None):

	return np.linalg.norm(r[:3]) - cst.d_M

def moon_SOI_reached(t, r, mass=None, T=None, eps=None):

	return abs(np.linalg.norm(r[:3]) - cst.d_M) - cst.SOI_M

def apside_pass(t, r, mass=None, T=None, eps=None):
	""" Determines if the S/C passes either at the perigee and/or apogee. Detected by the x-velocity component change of 
		sign ((+) -> (-) for apogee, (-) -> (+) for perigee)"""
	v_x = r[3]

	return v_x

def thrust_ignition(t, y, mass=None, T=None, eps=None):
	""" Raises an event each time the S/C pass a thrusters ignition or shut-down point """
	phi = np.arccos( np.dot( np.array([-1, 0, 0]), y[:3]) / np.linalg.norm(y[:3]) )

	return phi - eps

def cr3bp_moon_approach(t, r, cr3bp, mass, Tmax, thrusts_intervals=None):
	""" Raises an event when the S/C is at less than 30,000km of the Moon """

	# Moon's position in the synodic frame
	r_m = np.array([1 - cr3bp.mu, 0 , 0])
	d = np.linalg.norm(r[:3] - r_m)

	return d - 30000 / cr3bp.L

def angle_w_Ox(r):
	""" Return the (oriented) angle between the (Ox) axis and the `r` vector """
	return np.sign(np.cross(np.array([1, 0, 0]), r)[2]) * np.arccos( np.dot(np.array([1, 0, 0]), r) / np.linalg.norm(r) )

def plot_env_2D(ax):
	""" Plot the Earth, and the Moon's orbit on a pre-defined 2D-figure """

	ax.plot([0], [0], 'o', markersize=8, color='black', label='Earth')
	ax.plot([cst.d_M * np.cos(t_) for t_ in np.linspace(0, 2*np.pi, 1000)], [cst.d_M * np.sin(t_) for t_ in np.linspace(0, 2*np.pi, 1000)], '-', \
		linewidth=1, color='black', label='Moon trajectory')

def plot_env_3D(ax, earth_size=5):
	""" Plot the Earth, and the Moon's orbit on a pre-defined 2D-figure """

	ax.plot([0], [0], [0], 'o', markersize=earth_size, color='black', label='Earth')
	ax.plot([cst.d_M * np.cos(t_) for t_ in np.linspace(0, 2*np.pi, 1000)], [cst.d_M * np.sin(t_) for t_ in np.linspace(0, 2*np.pi, 1000)], \
		np.zeros(1000), '-', linewidth=0.5, color='black', alpha=0.8, label='Moon trajectory')


def thrust_profil_construction(time, trajectory, thrusts_intervals, Tmax):
	""" Converts a thrust interval matrix into a thrust profil assuming the thrust direction is following
		the S/C velocity """

	thrusts = np.empty((0, 4))

	for k, t in enumerate(time):
		ind = np.searchsorted(a=thrusts_intervals[:, 0], v=t, side='right') - 1

		if thrusts_intervals[ind, 1] >= t:
			velocity = trajectory[3:, k]
			ux, uy, uz = Tmax * velocity / np.linalg.norm(velocity)
			T = Tmax

			thrusts = np.vstack((thrusts, np.array([T, ux, uy, uz])))

		else:
			thrusts = np.vstack((thrusts, np.array([0, 0, 0, 0])))

	thrusts = np.transpose(thrusts)

	return thrusts

def thrust_index(time, thrust_profil, Tmax):

	index = list()

	ind_l, ind_u = 0, 0
	thrust_ON = False

	for k, t in enumerate(time):

		if (thrust_profil[0, k] >= 0.1 * Tmax and thrust_ON == False):
			ind_l = k
			thrust_ON = True 

		elif (thrust_profil[0, k] <= 0.1 * Tmax and thrust_ON == True):
			ind_u = k
			thrust_ON = False
			index.append([ind_l, ind_u])

		elif  (thrust_profil[0, k] >= 0.1 * Tmax and t == time[-1]):
			ind_u = k 
			thrust_ON = False
			index.append([ind_l, ind_u])

	return np.array(index)

def plot_earth_departure(apogee_raising_trajectory, apogee_raising_time, apogee_raising_thrust_interval, \
	moon_moon_trajectory, moon_moon_time, moon_moon_thrust_interval, outter_trajectory, outter_time, Tmax):

	r0 = apogee_raising_trajectory[:, -1]
	t_span = [0, 86400 * 1.5]
	t_eval = np.linspace(t_span[0], t_span[1], 10000)
	r0[3:] = [-1, -1, -0.2]

	propagation = solve_ivp(fun=kepler, y0=r0, t_span=t_span, t_eval=t_eval, atol=1e-13, rtol=1e-10)


	
	apogee_raising_thrusts = thrust_profil_construction(apogee_raising_time, apogee_raising_trajectory, apogee_raising_thrust_interval, Tmax)
	# moon_moon_thrusts = thrust_profil_construction(moon_moon_time, moon_moon_trajectory, moon_moon_thrust_interval, Tmax)

	apogee_raising_thrusts_index = thrust_index(apogee_raising_time, apogee_raising_thrusts, Tmax)
	# moon_moon_thrusts_index = thrust_index(moon_moon_time, moon_moon_thrusts, Tmax)

	apogee_raising_coasts_index = list()
	for k in range(len(apogee_raising_thrusts_index)-1):
		apogee_raising_coasts_index.append([apogee_raising_thrusts_index[k][1], apogee_raising_thrusts_index[k+1][0]])
	apogee_raising_coasts_index.append([apogee_raising_thrusts_index[-1][1], len(apogee_raising_trajectory[0])])

	apogee_raising_coasts_index = np.reshape(apogee_raising_coasts_index, (int(len(apogee_raising_coasts_index)), 2))

	fig = plt.figure()
	ax = fig.gca(projection='3d')


	ax.plot(moon_moon_trajectory[0], moon_moon_trajectory[1], moon_moon_trajectory[2], '-', color='blue', alpha=0.7, linewidth=1)
	# ax.plot(outter_trajectory[0], outter_trajectory[1], outter_trajectory[2], '-', color='blue', linewidth=1)

	for ind in apogee_raising_coasts_index:
		ax.plot(apogee_raising_trajectory[0, ind[0]:ind[1]], apogee_raising_trajectory[1, ind[0]:ind[1]], apogee_raising_trajectory[2, ind[0]:ind[1]], '-', color='blue', alpha=0.7, linewidth=1)


	for ind in apogee_raising_thrusts_index:
		ax.plot(apogee_raising_trajectory[0, ind[0]:ind[1]], apogee_raising_trajectory[1, ind[0]:ind[1]], apogee_raising_trajectory[2, ind[0]:ind[1]], '-', color='red', alpha=0.7, linewidth=1)

	ax.plot([apogee_raising_trajectory[0, -1]], [apogee_raising_trajectory[1, -1]], [apogee_raising_trajectory[2, -1]], marker='o', color=(0, 0, 1, 0), markeredgecolor='magenta', \
		markeredgewidth=1, label='1$^{st}$ - 2$^{nd}$ LGAs')

	ax.plot(propagation.y[0], propagation.y[1], propagation.y[2], '-', color='green', alpha=0.7, linewidth=1, label='Outbound trajectory')

	# for ind in moon_moon_thrusts_index:
	# 	ax.plot(moon_moon_trajectory[0, ind[0]:ind[1]], moon_moon_trajectory[1, ind[0]:ind[1]], moon_moon_trajectory[2, ind[0]:ind[1]], '-', color='red', linewidth=1)

	plot_env_3D(ax)

	ax.plot([0, 1], [0, 1], [0, 1], '-', color='red', alpha=0.7, label='Thrust')
	ax.plot([0, 1], [0, 1], [0, 1], '-', color='blue', alpha=0.7, label='Coast')

	ax.set_xlabel('X [km]')
	ax.set_ylabel('Y [km]')
	ax.set_zlabel('Z [km]')

	ax.set_xlim(-500000, 500000)
	ax.set_ylim(-500000, 500000)
	ax.set_zlim(-500000, 500000)

	plt.legend()
	plt.show()



	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(np.concatenate((apogee_raising_time, moon_moon_time))/86400, np.concatenate((apogee_raising_thrusts[0], np.zeros(len(moon_moon_time)))), color='blue')

	ax.set_xlabel('Time [d]')
	ax.set_ylabel('Thrust [N]')

	plt.grid()
	plt.show()
	
	




