import pykep as pk 

NAME2SPK = {'2020 CD3': '54000953',
			'2018 WV1': '3836309',
			'2017 KJ32': '3775115',
			'2014 WX202': '3698849',
			'2011 UD21': '3582145',
			'2008 JL24': '3410533',
			'1991 VG': '3005816',
			'2018 BC': '3797700',
			'2015 VO142': '3735181',
			'2014 WU200': '3697803',
			'2008 UA202': '3435539',
			'2006 RH120': '3403148',
			'2007 UN21': '2294227'}

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
	# Load SPICE kernels
	pk.util.load_spice_kernel('spice_kernels/asteroids.bsp')
	pk.util.load_spice_kernel('spice_kernels/de405.bsp')

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
	# Load SPICE kernels
	pk.util.load_spice_kernel('spice_kernels/de405.bsp')

	return pk.planet.spice(name, observer, ref_frame, aberrations, mu_central_body)