################################################################################
###                                 Imports                                  ###
################################################################################
#Standard imports
import math
import warnings

#Third party imports
import numpy as np

################################################################################
###                                Constants                                 ###
################################################################################
C_TO_KEL_OFFSET = 273.15

################################################################################
###                              Speed of Sound                              ###
################################################################################
def c_mackenzie(T, z, S=35):
	"""
	Computes the speed of sound underwater based on Mackenzie 1981. Only valid 
	for 2 < T < 30, 25 < S < 40, z < 8e3. Raises a warning if outside valid 
	limits
	http://resource.npl.co.uk/acoustics/techguides/soundseawater/underlying-phys.html

	Parameters
	----------
	T : float
		Water temperature in Celsius
	z : float
		Depth in meters
	S : float
		Salinity in parts per thousand (ppt)

	Returns
	-------
	c : float
		Speed of sound in meters per second
	"""
	#Make sure depth is non negative
	if z < 0:
		z = 0

	#Check limits
	if T < 2 or T > 30 or S < 25 or S > 40 or z > 8e3:
		warnings.warn("Inputs are outside reliability limits for Mackenzie")

	#Compute sound speed
	return 1448.96 + 4.591 * T - 5.304e-2 * (T ** 2) + 2.374e-4 * (T ** 3) + 1.340 * (S - 35) + 1.630e-2 * z + 1.675e-7 * (z ** 2) - 1.025e-2 * T * (S - 35) - 7.139e-13 * T * (z ** 3)

################################################################################
def c_coppens(T, z, S=35):
	"""
	Computes the speed of sound underwater based on Coppens 1981. Only valid 
	for 0 < T < 35, 0 < S < 45, z < 4e3. Raises a warning if outside valid 
	limits
	http://resource.npl.co.uk/acoustics/techguides/soundseawater/underlying-phys.html

	Parameters
	----------
	T : float
		Water temperature in Celsius
	z : float
		Depth in meters
	S : float
		Salinity in parts per thousand (ppt)

	Returns
	-------
	c : float
		Speed of sound in meters per second
	"""
	#Make sure depth is non negative
	if z < 0:
		z = 0

	#Check limits
	if T < 0 or T > 35 or S < 0 or S > 45 or z > 4e3:
		warnings.warn("Inputs are outside reliability limits for Coppens")

	#Convert depth to km for these calculations
	z = z / 1000

	#Compute sound speed
	t = T / 10
	c_surface = 1449.05 + 45.7 * t - 5.21 * (t ** 2) + 0.23 * (t ** 3) + (1.333 - 0.126 * t + 0.009 * (t ** 2)) * (S - 35)
	return c_surface + (16.23 + 0.253 * t) * z + (0.213 - 0.1 * t) * (z ** 2) + (0.016 + 0.0002 * (S - 35)) * (S - 35) * t * z

################################################################################
def speed_of_sound(T, z, S=35):
	"""
	Computes the speed of sound underwater. Uses various models and averages 
	all valid models together. If no models are valid for the given parameters 
	then it raises a warning

	Parameters
	----------
	T : float
		Water temperature in Celsius
	z : float
		Depth in meters
	S : float
		Salinity in parts per thousand (ppt)

	Returns
	-------
	c : float
		Speed of sound in meters per second
	"""
	valid_results = []
	invalid_results = []

	#Compute mackenzie
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		c = c_mackenzie(T, z, S)
		if w:
			invalid_results.append(c)
		else:
			valid_results.append(c)

	#Compute coppens
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		c = c_coppens(T, z, S)
		if w:
			invalid_results.append(c)
		else:
			valid_results.append(c)

	#Average valid results if we have any, if everything is invalid then 
	#average invalid results and warn
	c = 1500
	if valid_results:
		c = np.mean(np.array(valid_results))
	else:
		c = np.mean(np.array(invalid_results))
		warnings.warn("Inputs are outside reliability limits for all speed of sound models")

	#Return absorption
	return c

################################################################################
###                                Absorption                                ###
################################################################################
def alpha_fisher_and_simmons(freq, T, z):
	"""
	Absorption of sound in water based on Fisher and Simmons 1977. This is only 
	valid for Lyman and Fleming standard of water where Salinity = 35 part per 
	thousand (ppt) and Acidity = 8 pH

	Parameters
	----------
	freq : float
		Frequency of operation in Hz
	T : float
		Temperature of water in Celcius
	z : float
		Depth of water in meters. Must be non-negative
	
	Returns
	-------
	alpha : float
		Absorption in dB/km
	"""
	#Make sure depth is non negative
	if z < 0:
		z = 0

	#Convert temperature to kelvin
	T_kel = T + C_TO_KEL_OFFSET

	#Relate depth to pressure
	p = z / 10.0

	#Compute contribution from boric acid
	alpha = 1.03 * (10 ** -8) + 2.36 * (10 ** -10) * T - 5.22 * (10 ** -12) * (T ** 2)
	p_correction = 1
	f = 1.32 * (10 ** 3) * T_kel * math.exp(-1700 / T_kel)
	boric = (alpha * p_correction * f * (freq ** 2)) / ((freq ** 2) + (f ** 2))

	#Compute contribution from MgSO4
	alpha = 5.62 * (10 ** -8) + 7.52 * (10 ** -10) * T
	p_correction = 1 - 10.3 * (10 ** -4) * p + 3.7 * (10 ** -7) * (p ** 2)
	f = 1.55 * (10 ** 7) * T_kel * math.exp(-3052 / T_kel)
	mgso4 = (alpha * p_correction * f * (freq ** 2)) / ((freq ** 2) + (f ** 2))

	#Compute pure water contribution
	alpha = (55.9 - 2.37 * T + 4.77 * (10 ** -2) * (T ** 2) - 3.48 * (10 ** -4) * (T ** 3)) * (10 ** -15)
	p_correction = 1 - 3.84 * (10 ** -4) * p + 7.57 * (10 ** -8) * (p ** 2)
	h2o = alpha * p_correction * (freq ** 2)

	#Compute total absorption
	alpha = (boric + mgso4 + h2o) * 8686

	#Return absorption in dB/km
	return alpha

################################################################################
def alpha_francois_and_garrison(freq, T, z, S=35, ph=8):
	"""
	Absorption of sound in water based on Francois and Garrison 1982. This 
	model is estimated to be about 5% accurate. For frequencies between 10 and 
	500 kHz the MgSO4 contribution dominates and the limits of reliability are 
	as follows: -2 < T < 22, 30 < S < 35, 0 < z < 3500. For frequencies greater 
	than 500 kHz, the pure water contribution dominates and the limits of 
	reliability are: 0 < T < 30, 0 < S < 40, 0 < z < 10e3. Will raise a warning 
	if input goes outside of the limits

	Parameters
	----------
	freq : float
		Frequency of operation in Hz
	T : float
		Temperature of water in Celcius
	z : float
		Depth of water in meters. Must be non-negative
	S : float
		Salinity in parts per thousand (ppt)
	ph : float
		Acidity in pH
	
	Returns
	-------
	alpha : float
		Absorption in dB/km
	"""
	#Make sure depth is non negative
	if z < 0:
		z = 0

	#Check limits
	if freq < 10e3:
		warnings.warn("Inputs are outside reliability limits for Francois and Garrison")
	elif freq >= 10e3 and freq <= 500e3:
		if T < -2 or T > 22 or S < 30 or S > 35 or z > 3500:
			warnings.warn("Inputs are outside reliability limits for Francois and Garrison")
	else:
		if T < 0 or T > 30 or S < 0 or S > 40 or z > 10e3:
			warnings.warn("Inputs are outside reliability limits for Francois and Garrison")

	#Convert frequency to kHz for these calculations
	freq = freq / 1000

	#Convert temperature to kelvin
	T_kel = T + C_TO_KEL_OFFSET

	#Compute speed of sound according to Francois and Garrison
	c = 1412 + 3.21 * T + 1.19 * S + 0.0167 * z

	#Compute boric acid contribution
	alpha = (8.86 / c) * (10 ** (0.78 * ph - 5))
	p_correction = 1
	f = 2.8 * math.sqrt(S / 35) * (10 ** (4 - 1245 / T_kel))
	boric = (alpha * p_correction * f * (freq ** 2)) / ((freq ** 2) + (f ** 2))

	#Compute MgSO4 contribution
	alpha = 21.44 * (S / c) * (1 + 0.025 * T)
	p_correction = 1 - 1.37e-4 * z + 6.2e-9 * (z ** 2)
	f = (8.17 * (10 ** (8 - 1990 / T_kel))) / (1 + 0.0018 * (S - 35))
	mgso4 = (alpha * p_correction * f * (freq ** 2)) / ((freq ** 2) + (f ** 2))

	#Compute pure water contribution
	if T <= 20:
		alpha = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * (T ** 2) - 1.5e-8 * (T ** 3)
	else:
		alpha = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * (T ** 2) - 6.5e-10 * (T ** 3)
	p_correction = 1 - 3.83e-5 * z + 4.9e-10 * (z ** 2)
	h2o = alpha * p_correction * (freq ** 2)

	#Compute total absorption
	alpha = (boric + mgso4 + h2o)

	#Return absorption in dB/km
	return alpha

################################################################################
def alpha_ainslie_and_mccolm(freq, T, z, S=35, ph=8):
	"""
	Absorption of sound in water based on Ainslie and McColm 1998. This 
	model is estimated to be about 10% accurate for frequencies between 100 Hz 
	and 1 MHz. The limits of reliability are: -6 < T < 35, 7.7 < pH < 8.3, 
	5 < S < 50, and z < 7000. Will raise a warning if input goes outside of the 
	limits

	Parameters
	----------
	freq : float
		Frequency of operation in Hz
	T : float
		Temperature of water in Celcius
	z : float
		Depth of water in meters. Must be non-negative
	S : float
		Salinity in parts per thousand (ppt)
	ph : float
		Acidity in pH
	
	Returns
	-------
	alpha : float
		Absorption in dB/km
	"""
	#Make sure depth is non negative
	if z < 0:
		z = 0

	#Check limits
	if freq < 100:
		warnings.warn("Inputs are outside reliability limits for Ainslie and McColm")
	elif freq >= 100 and freq <= 1e6:
		if T < -6 or T > 35 or S < 5 or S > 50 or ph < 7.7 or ph > 8.3 or z > 7000:
			warnings.warn("Inputs are outside reliability limits for Ainslie and McColm")
	else:
		warnings.warn("Inputs are outside reliability limits for Ainslie and McColm")

	#Convert frequency to kHz for these calculations
	freq = freq / 1000

	#Convert depth to km for these calculations
	z = z / 1000

	#Convert temperature to kelvin
	T_kel = T + C_TO_KEL_OFFSET

	#Compute boric acid contribution
	alpha = 0.106 * math.exp((ph - 8) / (0.56))
	p_correction = 1
	f = 0.78 * math.sqrt(S / 35) * math.exp(T / 26)
	boric = (alpha * p_correction * f * (freq ** 2)) / ((freq ** 2) + (f ** 2))

	#Compute MgSO4 contribution
	alpha = 0.52 * (S / 35) * (1 + T / 43)
	p_correction = math.exp(-z / 6)
	f = 42 * math.exp(T / 17)
	mgso4 = (alpha * p_correction * f * (freq ** 2)) / ((freq ** 2) + (f ** 2))

	#Compute pure water contribution
	alpha = 0.00049 * math.exp(-(T / 27 + z / 17))
	p_correction = 1
	h2o = alpha * p_correction * (freq ** 2)

	#Compute total absorption
	alpha = (boric + mgso4 + h2o)

	#Return absorption in dB/km
	return alpha

################################################################################
def absorption(freq, T, z, S=35, ph=8):
	"""
	Absorption of sound in water. Uses 3 different models: 
	Fisher and Simmons 1977, Francois and Garrison 1982, and Ainslie and 
	McColm 1998. Averages all valid models together. If no models are valid for 
	the given parameters then it raises a warning

	Parameters
	----------
	freq : float
		Frequency of operation in Hz
	T : float
		Temperature of water in Celcius
	z : float
		Depth of water in meters. Must be non-negative
	S : float
		Salinity in parts per thousand (ppt)
	ph : float
		Acidity in pH
	
	Returns
	-------
	alpha : float
		Absorption in dB/km
	"""
	valid_results = []
	invalid_results = []

	#Compute fisher and simmons which is only valid for S = 35 and pH = 8
	alpha = alpha_fisher_and_simmons(freq, T, z)
	if S == 35 and ph == 8:
		valid_results.append(alpha)
	else:
		invalid_results.append(alpha)

	#Compute francois and garrison
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		alpha = alpha_francois_and_garrison(freq, T, z, S=S, ph=ph)
		if w:
			invalid_results.append(alpha)
		else:
			valid_results.append(alpha)

	#Compute ainslie and mccolm
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		alpha = alpha_ainslie_and_mccolm(freq, T, z, S=S, ph=ph)
		if w:
			invalid_results.append(alpha)
		else:
			valid_results.append(alpha)

	#Average valid results if we have any, if everything is invalid then 
	#average invalid results and warn
	alpha = 0
	if valid_results:
		alpha = np.mean(np.array(valid_results))
	else:
		alpha = np.mean(np.array(invalid_results))
		warnings.warn("Inputs are outside reliability limits for all absorption models")

	#Return absorption
	return alpha

################################################################################
###                            Transmission Loss                             ###
################################################################################
def absorption_loss(d, alpha):
	"""
	Computes the loss due to absorption for a specific distance or set of 
	distances. Return values are loss values meaning they are greater than 1 
	linear. You will need to flip them to make the gain values

	Parameters
	----------
	d : float or ndarray
		Single distance or a vector of distances in meters
	alpha : float
		Absorption coefficient in dB/km

	Returns
	-------
	lin_one_way : float or ndarray
		One way loss due to absorption as a linear value
	lin_two_way : float or ndarray
		Two way loss due to absorption as a linear value
	db_one_way : float or ndarray
		One way loss due to absorption in dB
	db_two_way : float or ndarray
		Two way loss due to absorption in dB
	"""
	#Convert list to numpy array if input is list
	if type(d) is list:
		d = np.array(d)

	#Change alpha to dB/m
	alpha = alpha / 1000

	#Calculate absorption loss
	db_one_way = alpha * d
	db_two_way = db_one_way * 2
	lin_one_way = 10 ** (db_one_way / 10)
	lin_two_way = 10 ** (db_two_way / 10)

	#Return loss in various formats
	return lin_one_way, lin_two_way, db_one_way, db_two_way

################################################################################
def spreading_loss(d):
	"""
	Computes the loss due to spherical spreading for a specific distance or 
	distances. Simply uses 20 * log10(d) for one way loss. Return values are 
	loss values meaning they are greater than 1 linear. You will need to flip 
	them to make the gain values. Simplified equation only works for ranges 1 
	meter or greater so any ranges less than 1 meter will be rounded up to 
	1 meter.

	Parameters
	----------
	d : float or ndarray
		Single distance or a vector of distances in meters

	Returns
	-------
	lin_one_way : float or ndarray
		One way loss due to spherical spreading as a linear value
	lin_two_way : float or ndarray
		Two way loss due to spherical spreading as a linear value
	db_one_way : float or ndarray
		One way loss due to spherical spreading in dB
	db_two_way : float or ndarray
		Two way loss due to spherical spreading in dB
	"""
	#Convert list to numpy array if input is list
	if type(d) is list:
		d = np.array(d)

	#Check if d is single value or array
	try:
		d.shape
		#If made it here then d is an array so round any values less than 1 up 
		#to 1
		d[np.where(d < 1)] = 1
	except Exception as e:
		#If made it here then d is not an array so it must be a single value
		#Round up to 1 if less than 1
		if d < 1:
			d = 1

	#Calculate spreading loss
	db_one_way = 20 * np.log10(d)
	db_two_way = db_one_way * 2
	lin_one_way = 10 ** (db_one_way / 10)
	lin_two_way = 10 ** (db_two_way / 10)

	#Return loss in various formats
	return lin_one_way, lin_two_way, db_one_way, db_two_way

################################################################################
def transmission_loss(d, alpha):
	"""
	Computes the loss due to absorption and spherical spreading for a specific 
	distance or set of distances. Return values are loss values meaning they 
	are greater than 1 linear. You will need to flip them to make the gain 
	values. Simplified equation only works for ranges 1 meter or greater so any 
	ranges less than 1 meter will be rounded up to 1 meter.

	Parameters
	----------
	d : float or ndarray
		Single distance or a vector of distances in meters
	alpha : float
		Absorption coefficient in dB/km

	Returns
	-------
	lin_one_way : float or ndarray
		One way loss due to absorption as a linear value
	lin_two_way : float or ndarray
		Two way loss due to absorption as a linear value
	db_one_way : float or ndarray
		One way loss due to absorption in dB
	db_two_way : float or ndarray
		Two way loss due to absorption in dB
	"""
	#Calculate transmission loss
	_, _, absorption_db_one_way, _ = absorption_loss(d, alpha)
	_, _, spreading_db_one_way, _ = spreading_loss(d)
	db_one_way = absorption_db_one_way + spreading_db_one_way
	db_two_way = db_one_way * 2
	lin_one_way = 10 ** (db_one_way / 10)
	lin_two_way = 10 ** (db_two_way / 10)

	#Return loss in various formats
	return lin_one_way, lin_two_way, db_one_way, db_two_way

################################################################################
###                                Test Code                                 ###
################################################################################
if __name__ == "__main__":
	pass

################################################################################
###                               End of File                                ###
################################################################################