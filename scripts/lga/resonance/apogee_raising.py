import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scripts.lga.resonance.utils_2 import kepler_thrust, kep2cart, moon_reached

"""
	This script deals with the study of the orbit raising using Oberth effect to reach the Moon before
	a LGA.

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


# 2 - Definition of the problem parameters
# ----------------------------------------

# S/C maximal thrust [kN]
T = 1e-3

# Angle allowing the thrusters ignition or shutdown [rad]
eps = 2 * np.pi / 180


# 3 - Definition of the initial orbit parameters
# ----------------------------------------------
a  = 2 * (R_E + 200)        # SMA [km]
e  = 1 - (R_E + 200) / a    # Eccentricity [-]
i  = 0                      # Inclinaison [rad]
W  = 0                      # RAAN [rad]
w  = np.pi                  # Perigee anomaly [rad]
ta = 0                      # True anomaly [rad]


# 4 - Propagation of the Keplerian equations
# ------------------------------------------
r0 = kep2cart(a, e, i, W, w, ta, mu_E) 

t_span = np.array([0, 100 * 86400])
t_eval = np.linspace(t_span[0], t_span[-1], 10000)

moon_reached.terminal = True
moon_reached.direction = 1
sol = solve_ivp(fun=kepler_thrust, y0=r0, t_span=t_span, t_eval=t_eval, args=(T, eps), events=moon_reached, rtol=1e-12, atol=1e-13)
r = sol.y

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot([0], [0], [0], 'o', color='black', markersize=7, label='Earth')
ax.plot(r[0], r[1], r[2], '-', color='blue', linewidth=1, label='S/C trajectory')

ax.plot( [d_M*np.cos(theta_) for theta_ in np.linspace(0, 2*np.pi, 100)], [d_M*np.sin(theta_) for theta_ in np.linspace(0, 2*np.pi, 100)], \
			np.zeros(100), '--', color='black', linewidth=1, label='Moon orbit')

plt.legend()
plt.show()



