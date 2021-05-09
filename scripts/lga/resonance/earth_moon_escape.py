import numpy as np 

"""
	In this script, we deal with the case where a trajectory between an elliptic orbit around the Earth
	intercepting the Moon's orbit with a velocity at infinity before the LGA equal to the velocity at 
	infinity after the LGA.

	We define a cartesian and a spherical coordinate systems around the Moon and determine if a simple LGA
	is sufficient or if a second one is necessary.

	Parameters:
	-----------

		v_inf_mag : [km/s]
			Velocity at infinity before and after the LGA 
		phi_m, theta_m : [rad]
			Azimuthal and polar angles of the velocity at infinity vector before the LGA
		phi_p, theta_p : [rad]
			Azimuthal and polar angles of the velocity at infinity vector after the LGA
		r_p : [km]
			Minimal distance between the S/C and the Moon surface during the encounter

"""

# 1 - Definition of the problem parameters
# ----------------------------------------

# Velocity at infinity before and after the LGA [km/s]
v_inf_mag = 1.0

# Azimuthal and polar angles of the velocity at infinity vector before the LGA [rad]
phi_m = 0 * np.pi / 180
theta_m = 0 * np.pi / 180

# Azimuthal and polar angles of the velocity at infinity vector after the LGA [rad]
phi_p = 0 * np.pi / 180
theta_p = 0 * np.pi / 180

# Minimal distance between the S/C and the Moon surface during the encounter [km]
r_m = 100


# 2 - Definition of Moon's characteristics
# ----------------------------------------

# Moon's gravitational parameter [km^3/s^2]
mu_M = 4902.7779

# Moon's radius [km]
R_M = 1737.4

# Moon's orbital velocity [km/s]
V_M = 1.022


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

# S/C velocity at infinity after the LGA [km/s]
v_inf_p = v_inf_mag * np.array([ np.cos(phi_p) * np.sin(theta_p),
								 np.sin(phi_p) * np.sin(theta_p),
								 np.cos(theta_p)                ])


# 5 - Computation of the angle between the velocities at infinity before and after the LGA and comparaison with the maximal rotation [rad]
# ----------------------------------------------------------------------------------------------------------------------------------------

# Angle between the velocities at infinity before and after the LGA [rad]
delta = np.arccos( np.dot(v_inf_m, v_inf_p) / v_inf_mag**2 )

# Maximum rotation angle [rad]
delta_m = 2 * np.arcsin( (R_M+r_m) / mu_M / (2 * v_inf_mag**2 + (R_M+r_m) / mu_M) )

print("Delta     : {}°".format(delta   * 180 / np.pi))
print("Delta max : {}°".format(delta_m * 180 / np.pi))

if abs(delta) > abs(delta_m):
	print("\n\tA second LGA is necessary.")
else:
	print("\n\tOne LGA is sufficient.")