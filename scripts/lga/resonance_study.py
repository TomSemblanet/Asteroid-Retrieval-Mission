import matplotlib.pyplot as plt
import numpy as np 


"""
	In this script, we deal with the case where 1 LGA isn't sufficient and where our aim is to enter in resonance
	with the Moon to attempt a 2nd LGA.

	We define a p:q resonance ratio and try to find a suitable couple (phi_p, theta_p) allowing our S/C to catch
	the Moon a second time after p revolutions.

	Parameters:
	-----------

		v_inf_mag : [km/s]
			Velocity at infinity before and after the LGA 
		phi_m, theta_m : [rad]
			Azimuthal and polar angles of the velocity at infinity vector before the LGA
		r_p : [km]
			Minimal distance between the S/C and the Moon surface during the encounter
		p : [-]
			Number of revolution the S/C will accomplish before 2nd Moon encouter
		q : [-]
			Number of revolution the Moon will accomplish before 2nd S/C encouter

"""

# 1 - Definition of the problem parameters
# ----------------------------------------

# Velocity at infinity before and after the LGA [km/s]
v_inf_mag = 1

# Azimuthal and polar angles of the velocity at infinity vector before the LGA [rad]
phi_m = 30 * np.pi / 180
theta_m = 40 * np.pi / 180

# Minimal distance between the S/C and the Moon surface during the encounter [km]
r_m = 100

# Number of revolution the S/C will accomplish before 2nd Moon encouter
p = 1

# Number of revolution the Moon will accomplish before 2nd S/C encouter
q = 1


# 2 - Definition of Moon's and Earth's characteristics
# ----------------------------------------------------

# Earth's gravitational parameter [km^3/s^2]
mu_E = 398600.4418

# Earth-Moon mean distance [km]
d_M = 385000

# Moon's gravitational parameter [km^3/s^2]
mu_M = 4902.7779

# Moon's radius [km]
R_M = 1737.4

# Moon's orbital velocity [km/s]
V_M = 1.022

# Moon's orbit period [s]
T_M = 2551392


# 3 - Definition of the frame centered on the Moon
# ------------------------------------------------

# i : Angular momentum of the Moon (normalized) [-]
i = np.array([1, 0, 0])

# j : Position vector of the Moon relative to the Earth (normalized) [-]
j = np.array([0, 1, 0])

# k : Velocity vector of the Moon relative to the Earth (normalized) [-]
k = np.array([0, 0, 1])


# 4 - Definition of the velocity vector of the Moon and S/C velocity at infinity before/after the LGA in the Moon centered frame [km/s]
# -------------------------------------------------------------------------------------------------------------------------------------

# Moon's velocity vector [km/s]
v_M = np.array([0, 0, V_M])

# S/C velocity at infinity before the LGA [km/s]
v_inf_m = v_inf_mag * np.array([ np.cos(phi_m) * np.sin(theta_m),
								 np.sin(phi_m) * np.sin(theta_m),
								 np.cos(theta_m)                ])


# 5 - Computation of the polar angle of the S/C velocity at infinity after LGA to enter in p:q resonance with the Moon [rad]
# --------------------------------------------------------------------------------------------------------------------------

# S/C velocity after LGA to enter in a p:q resonance with the Moon [km/s]
v = np.sqrt( 2*mu_E/d_M - (2*np.pi * mu_E / (T_M * p/q))**(2/3) )

# Polar angle of the S/C velocity at infinity after LGA [rad]
theta_p = np.pi - np.arccos( (v**2 - V_M**2 - v_inf_mag**2) / (2 * (V_M**2) * (v_inf_mag**2)) )


# 6 - Computation of the admissible longitude angles of the S/C velocity at infinity after LGA to enter in p:q resonance with the Moon [rad]
# ------------------------------------------------------------------------------------------------------------------------------------------

# Computation of the maximum rotation [rad]
delta_max = 2 * np.arcsin( mu_M/(R_M+r_m) / (v_inf_mag**2 + mu_M/(R_M+r_m)) )

# Possible longitude angles [rad]
phi_p_arr = np.linspace(-np.pi, np.pi, 100)

# Admissible longitude angles [rad]
phi_p_adm = np.array([])

def admissible_longitude(phi_p):
	return (np.cos(phi_m)*np.sin(theta_m)*np.sin(theta_p))*np.cos(phi_p) + \
		   (np.sin(phi_m)*np.sin(theta_m)*np.sin(theta_p))*np.sin(phi_p) + \
			np.cos(theta_m)*np.cos(theta_p) - np.cos(delta_max)

for phi_p in phi_p_arr:
	if admissible_longitude(phi_p) <= 0:
		phi_p_adm = np.append(phi_p_adm, phi_p)

print(phi_p_adm)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(phi_p_arr, np.array([ (np.cos(phi_m)*np.sin(theta_m)*np.sin(theta_p))*np.cos(phi_p) + \
					      	  (np.sin(phi_m)*np.sin(theta_m)*np.sin(theta_p))*np.sin(phi_p) + \
					    	  np.cos(theta_m)*np.cos(theta_p) - np.cos(delta_max) for phi_p in phi_p_arr]))

plt.grid()
plt.show()




	