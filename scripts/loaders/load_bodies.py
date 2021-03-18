import pykep as pk 

from data.spk_table import NAME2SPK


def load_asteroid(name, observer='SUN', ref_frame='ECLIPJ2000', aberrations='NONE', mu_central_body=pk.MU_SUN):
	""" Load an asteroid from its generic name (eg. '2011 UD21') using PyKEP.util
		module and SPICE kernels. The asteroid

		Parameters:
		-----------
		name: str
			Name of the asteroid 
		observer: str
			Name of the observer celestial body
		ref_fram: str
			Name of the referential frame 
		aberrations: str
			Aberration correction type
		mu_central_body: float
			Gravity parameter of the central body


		Returns:
		--------
		asteroid: <pykep.planet>
			Asteroid object

	"""
	return pk.planet.spice(NAME2SPK[name], observer, ref_frame, aberrations, mu_central_body)

def load_planet(name, observer='SUN', ref_frame='ECLIPJ2000', aberrations='NONE', mu_central_body=pk.MU_SUN):
	""" Load an asteroid from its generic name (eg. '2011 UD21') using PyKEP.util
		module and SPICE kernels. The asteroid

		Parameters:
		-----------
		name: str
			Name of the planet 
		observer: str
			Name of the observer celestial body
		ref_fram: str
			Name of the referential frame 
		aberrations: str
			Aberration correction type
		mu_central_body: float
			Gravity parameter of the central body


		Returns:
		--------
		asteroid: <pykep.planet>
			Planet object

	"""
	return pk.planet.spice(name, observer, ref_frame, aberrations, mu_central_body)