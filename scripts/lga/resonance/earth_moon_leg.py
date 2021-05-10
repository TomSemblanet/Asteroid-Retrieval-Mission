import numpy as np 
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.lga.resonance.utils_2 import kep2cart, kepler, moon_reached, R2

""" 
	In this script, we deal with the transfer from the Earth to the Moon with a highly elliptical trajectory
	reaching a given velocity at infinity at Moon arrival.

"""

# 1 - Definition of Moon's and Earth's characteristics
# ----------------------------------------------------

# Earth's gravitational parameter [km^3/s^2]
mu_E = 398600.4418

# Moon's gravitational parameter [km^3/s^2]
mu_M = 4902.7779

# Earth's radius [km]
R_E = 6384.415

# Moon's radius [km]
R_M = 1737.4

# Earth-Moon mean distance [km]
d_M = 385000

# Moon's orbital velocity [km/s]
V_M = 1.01750961806616

# Moon's orbit period [s]
T_M = 2377399

# 2 - Definition of problem parameters
# ------------------------------------
n_a = 10 # Number of SMA values
n_r = 10 # Number of perigee values

r_p_min = R_E + 200   # Minimum perigee radius [km]
r_p_max = R_E + 20000 # Maximum perigee radius [km]

a_max = 400000 # Maximum SMA [km]

# Possible S/C orbit around the Earth perigee [km]
r_p_arr = np.linspace(r_p_min, r_p_max, n_r)

# Storage array
delta_Vs = np.zeros(shape=(n_r, n_a))

for k, r_p in enumerate(r_p_arr):

	# Possible S/C orbit around the Earth SMA [km]
	a_arr = np.linspace((r_p+d_M)/2 + 100, a_max, n_a)

	# Values temporary storage
	buffer_ = np.array([])
		
	for a in a_arr:

		# 2 - Definition of the orbital parameters
		# ----------------------------------------
		a  = a            # SMA [km]
		e  = 1 - r_p / a  # Eccentricity [-]
		i  = 0            # Inclinaison [rad]
		W  = 0            # RAAN [rad]
		w  = np.pi        # Perigee anomaly [rad]
		ta = 0            # True anomaly [rad]


		# 3 - Conversion into cartesian parameters
		# ----------------------------------------
		r0 = kep2cart(a, e, i, W, w, ta, mu_E) 


		# 4 - Propagation of the keplerian equations
		# ------------------------------------------
		T = 2 * np.pi * np.sqrt(a**3/mu_E)
		t_span = [0, T]

		moon_reached.terminal = True
		# moon_reached.direction = -1
		moon_reached.direction = 1

		sol = solve_ivp(fun=kepler, t_span=t_span, y0=r0, events=moon_reached, rtol=1e-12, atol=1e-12)
		r = sol.y

		# S/C velocity vector at Moon interception [km/s]
		v_f = r[3:, -1]


		# 5 - Comparaison with Moon's velocity
		# ------------------------------------
		# Angle between (Ox) and the S/C position at t=tf
		theta = np.sign(np.cross(np.array([1, 0, 0]), r[:3, -1])[2]) * np.arccos(np.dot(np.array([1, 0, 0]), r[:3, -1]) / np.linalg.norm(r[:3, -1]))

		# Computation of the velocities difference [km/s]
		v_M = R2(theta).dot(np.array([0, V_M, 0]))
		delta_V = v_f - v_M

		buffer_ = np.append(buffer_, np.linalg.norm(delta_V))

	delta_Vs[k] = buffer_

xx, yy = np.meshgrid(a_arr, r_p_arr)

plt.contourf(xx, yy, delta_Vs, levels=500, cmap='viridis')  

plt.xlabel("SMA [km]")
plt.ylabel("Perigee radius [km]")

plt.title("Velocity at infinity at Moon arrival [km/s]")

plt.colorbar()  
plt.show()





