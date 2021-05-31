import math
import numpy as np

from scipy.optimize import minimize_scalar


def u(s):
	if np.linalg.norm(s) > 0 and np.linalg.norm(s) < 1:
		result = 1.5*np.linalg.norm(s)**3 - 2.5*np.linalg.norm(s)**2 + 1
	elif np.linalg.norm(s) > 1 and np.linalg.norm(s) < 2:
		result = -0.5*np.linalg.norm(s)**3 + 2.5*np.linalg.norm(s)**2 - 4*np.linalg.norm(s) + 2
	elif s==0:
		result = 1
	else:
		result = 0

	return result

def u_stept(s):
	return 0.5 - np.arctan(1000*(s-1))/np.pi

def u_tan(s):
	s_n = np.linalg.norm(s)

	a1 = u_stept(s_n)
	a2 = u_stept(s_n-1) - u_stept(s_n)

	return a1*(1.5*s_n**3 - 2.5*s_n**2 + 1) + a2*(-0.5*s_n**3 + 2.5*s_n**2 - 4*s_n + 2)

def step(s):
    return 1/(1+1000*s**(16))

def u_ad(s):
    s_n = np.linalg.norm(s)

    a = step(s_n-0.5)
    b = step(s_n - 1.5)

    return a*(1.5*s_n**3 - 2.5*s_n**2 + 1) + b*(-0.5*s_n**3 + 2.5*s_n**2 - 4*s_n + 2)


################################################################################

# Definition of the interpolated states of the manifold
def compute_x_ipl(grid, theta, tau, t1, t2, nb):

	# Note: the parameter nb in ranging from 0 to 5 and give the interpolation for respectively
	#		X, y, z, vx, vy, vz

	# Computing useful constants
	N1 = len(theta)
	N2 = len(tau)

	T10 = theta[0]
	T1 = theta[-1]

	T20 = tau[0]
	T2 = tau[-1]

	h1 = (T1-T10)/(N1-1)
	h2 = (T2-T20)/(N2-1)

	# Indices for summation
	idx = [-1, 0, 1, 2]

	# Finding the i and j indices to use in the computation 
	i = np.searchsorted(theta, t1, side = 'right') - 1
	j = np.searchsorted(tau, t2, side = 'right') - 1

	result = 0

	for l in idx:

		# u1 = u_tan((t1 - theta[i+l])/h1)
		u1 = u_tan((t1 - theta[i+l])/h1)

		for m in idx:

			# u2 = u_tan((t2 - tau[j+m])/h2)
			u2 = u_tan((t2 - tau[j+m])/h2)
			result += interp_coef(i+l,j+m,grid,nb)*u1*u2

	return result


# Function to compute inerpolation coefficients
def interp_coef(i,j,grid, nb):
	N1 = len(grid)
	N2 = len(grid[0].state_vec[:,0])

	# All the 'if' instructions are messing with automatic differentiation, 
	# Since it concerns boundary terms only here, we comment the related part

	if i==0:
		
		if j==0:

			c = 3*interp_coef(1,0,grid,nb) - 3*interp_coef(2,0,grid,nb) + interp_coef(3,0,grid,nb)
			return c

		elif j==N2+1:

			c = 3*interp_coef(1,N2+1,grid,nb) - 3*interp_coef(2,N2+1,grid,nb) + interp_coef(3,N2+1,grid,nb)
			return c

		else :
			# c = 3*grid[0].state_vec[j-1,nb] - 3*grid[1].state_vec[j-1,nb] + grid[2].state_vec[j-1,nb]
			c = 3*grid[0].state_vec[j,nb] - 3*grid[1].state_vec[j,nb] + grid[2].state_vec[j,nb]
			return c

	elif j==0:
		
		if i==N1+1:

			c = 3*interp_coef(N1,0,grid,nb) - 3*interp_coef(N1-1,0,grid,nb) + interp_coef(N1-2,0,grid,nb)
			return c

		else :
			# c = 3*grid[i-1].state_vec[0,nb] - 3*grid[i-1].state_vec[1,nb] + grid[i-1].state_vec[2,nb]
			c = 3*grid[i].state_vec[0,nb] - 3*grid[i].state_vec[1,nb] + grid[i].state_vec[2,nb]
			return c

	elif i==N1+1:
		if j == N2+1:

			c = 3*interp_coef(N1,N2+1,grid,nb) - 3*interp_coef(N1-1,N2+1,grid,nb) + interp_coef(N1-2,N2+1,grid,nb)
			return c

		else:

			c = 3*grid[N1-1].state_vec[j-1,nb] - 3*grid[N1-2].state_vec[j-1,nb] + grid[N1-3].state_vec[j-1,nb]
			c = 3*grid[N1-1].state_vec[j,nb] - 3*grid[N1-2].state_vec[j,nb] + grid[N1-3].state_vec[j,nb]
			return c

	elif j == N2+1:
		
		# c = 3*grid[i-1].state_vec[N2-1,nb] - 3*grid[i-1].state_vec[N2-2,nb] + grid[i-1].state_vec[N2-3,nb]
		c = 3*grid[i].state_vec[N2-1,nb] - 3*grid[i].state_vec[N2-2,nb] + grid[i].state_vec[N2-3,nb]
		return c

	else:
	
		c = grid[i].state_vec[j,nb]
	return c

# Function returning the Jacobi integral of motion
def compute_jacobi_int(vect, cr3bp):

	x = vect[0]
	y = vect[1]
	z = vect[2]

	vx = vect[3]
	vy = vect[4]
	vz = vect[5]

	mu = cr3bp.mu

	r1_temp = [x+mu,y,z]
	r2_temp = [x+mu-1,y,z]

	r1 = np.linalg.norm(r1_temp)
	r2 = np.linalg.norm(r2_temp)

	Jac = 2*(1-mu)/r1 + 2*mu/r2 + (x**2 + y**2) - (vx**2 + vy**2 + vz**2) + mu*(1-mu)

	return Jac

# Function that computes de derivative of the Jacobi integral of motion
def compute_Jx(vect_, cr3bp):

	x = vect_[0]
	y = vect_[1]
	z = vect_[2]

	vx = vect_[3]
	vy = vect_[4]
	vz = vect_[5]

	mu = cr3bp.mu

	r1_temp = [x+mu,y,z]
	r2_temp = [x+mu-1,y,z]

	r1 = np.linalg.norm(r1_temp)
	r2 = np.linalg.norm(r2_temp)

	Jx_1 = -2*(1-mu)*(x+mu)/(r1**3) - 2*mu*(x+mu-1)/(r2**3) + 2*x
	Jx_2 = -2*(1-mu)*y/(r1**3) - 2*mu*y/(r2**3) + 2*y
	Jx_3 = -2*(1-mu)*z/(r1**3) - 2*mu*z/(r2**3) 

	Jx_4 = -2*vx
	Jx_5 = -2*vy
	Jx_6 = -2*vz

	Jx = np.array([Jx_1, Jx_2, Jx_3, Jx_4, Jx_5, Jx_6])

	return Jx


# Function to compute the n vector, needed for the manifold approximation
def compute_n(ipl_states, crtbp):

	J_x = compute_Jx(ipl_states, crtbp)

	n = J_x/np.linalg.norm(J_x)

	return n

# Function computing delta, necessary for manifold approximation
def compute_delta(x_ipl, cr3bp, C_jac, n):

	delta = 0 
	eps = 1		# error 

	# # Newton method
	# while eps>10**(-14) and k<500:

	# 	k+=1

	# 	J = compute_jacobi_int(x_ipl + delta*n, cr3bp)
	# 	J_x = compute_Jx(x_ipl + delta*n, cr3bp)

	# 	norm_J_x = np.linalg.norm(J_x)

	# 	former_delta = delta 	# temprary variable to compute the error eps

	# 	delta -= (J - C_jac)/norm_J_x

	# 	eps = np.linalg.norm(former_delta - delta)

	for k in range(15):

		J = compute_jacobi_int(x_ipl + delta*n, cr3bp)
		J_x = compute_Jx(x_ipl + delta*n, cr3bp)

		norm_J_x = np.linalg.norm(J_x)

		former_delta = delta 	# temprary variable to compute the error eps

		delta -= (J - C_jac)/norm_J_x

		eps = np.linalg.norm(former_delta - delta)

	# def Jacobi_delta_scaled(delta):

	# 	return np.linalg.norm(compute_jacobi_int(x_ipl + delta*n, cr3bp) - C_jac)

	# result = minimize_scalar(Jacobi_delta_scaled)

	# delta = result.x

	return delta


####################################################################################################################################################
# Function to approximate the states of a manifold for two given parameters theta and tau
####################################################################################################################################################

def manifold_approximation(cr3bp, C_jac, grid, theta_array, tau_array, theta, tau):

	### Parameters ###
	# cr3bp : Cr3bp object in which we work (usually Earth-Moon)
	#
	# C_jac : Jacobi constant of the manifold we are considering
	#
	# grid : It is in fact the result of numerical integration of manifold, parametrized by theta and tau (= manifold[stable_interior] for ex)
	#
	# theta_array = array of discretized values of theta 
	#
	# tau_array = array of discretized values of tau
	#
	# theta : Parameter (along the orbit) 
	#
	# tau : Parameter (along the flow)
	#
	### Result ###
	#
	# x_app : approximated manifold at theta and tau

	# We build the kernel interpolation once
	#kernel = u_approx()

	if type(theta) != np.float64:
		theta = theta.value()
	if type(tau) != np.float64:
		tau = tau.value()

	# We begin by the interpolation of each component:
	x_ipl_x = compute_x_ipl(grid, theta_array, tau_array, theta, tau, 0)
	x_ipl_y = compute_x_ipl(grid, theta_array, tau_array, theta, tau, 1)
	x_ipl_z = compute_x_ipl(grid, theta_array, tau_array, theta, tau, 2)
	x_ipl_vx = compute_x_ipl(grid, theta_array, tau_array, theta, tau, 3)
	x_ipl_vy = compute_x_ipl(grid, theta_array, tau_array, theta, tau, 4)
	x_ipl_vz = compute_x_ipl(grid, theta_array, tau_array, theta, tau, 5)

	# The total interpolated function
	x_ipl = np.array([x_ipl_x, x_ipl_y, x_ipl_z, x_ipl_vx, x_ipl_vy, x_ipl_vz])

	# Compute intermediate objects ti build the manifold approx at the end
	n = compute_n(x_ipl, cr3bp)
	delta = compute_delta(x_ipl, cr3bp, C_jac, n)

	# Finally computing the approximated manifold
	x_app = x_ipl + delta*n

	return x_app


	