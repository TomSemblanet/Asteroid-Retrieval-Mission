import pygmo as pg

def load(name):
	""" Loads an SQP algorithm (either SLSQP or IPOPT) and return it as 
		a UDA (User-Defined Algorithm) usable by pygmo.algorithm method

		Parameters:
		-----------
		name: string
			Name of the SQP algorithm

		Returns:
		--------
		algo: <pygmo.uda>	
			SQP algorithm as a pygmo UDA

	"""
	
	if name == "slsqp":
		uda = pg.nlopt('slsqp')
		uda.xtol_rel = 0
		uda.ftol_rel = 0
		uda.maxeval = 200
		algo = pg.algorithm(uda)
		algo.set_verbosity(1)
		return algo

	elif name == "ipopt":
		uda = pg.ipopt() 
		uda.set_integer_option("print_level", 5)
		uda.set_integer_option("acceptable_iter", 4)
		uda.set_integer_option("max_iter", 500)

		uda.set_numeric_option("tol", 1e-5)
		uda.set_numeric_option("dual_inf_tol", 1e-5)
		uda.set_numeric_option("constr_viol_tol", 1e-5)
		uda.set_numeric_option("compl_inf_tol", 1e-5)

		uda.set_numeric_option("acceptable_tol", 1e-3)
		uda.set_numeric_option("acceptable_dual_inf_tol", 1e-2)
		uda.set_numeric_option("acceptable_constr_viol_tol", 1e-5)
		uda.set_numeric_option("acceptable_compl_inf_tol", 1e-5)

		algo = pg.algorithm(uda)
		return algo