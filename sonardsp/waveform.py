################################################################################
###                                 Imports                                  ###
################################################################################
#Standard imports
import math

#Third party imports
import numpy as np
import scipy.signal as signal

################################################################################
###                           Waveform Generation                            ###
################################################################################
def gen_lfm_chirp(fs, pw, fstart, fstop):
	"""
	Generates a complex Linear Frequency Modulated (LFM) waveform otherwise 
	known as a chirp

	Parameters
	----------
	fs : float
		Sampling frequency in Hz
	pw : float
		Pulse width in seconds, if pulse width is a fractional amount of 
		samples then the fraction will be truncated so the number of samples 
		will be rounded down
	fstart : float
		Start frequency in Hz, can be positive or negative
	fstop : float
		Stop frequency in Hz, can be positive or negative

	Returns
	-------
	t : ndarray of length floor(pw * fs) of type float64
		Time vector (in seconds) that goes with the chirp
	chirp : ndarray of length floor(pw * fs) of type complex128
		LFM waveform
	"""
	#Got chirp equation from here: https://en.wikipedia.org/wiki/Chirp
	n = math.floor(pw * fs)
	pw = n / fs
	t = np.arange(n) / fs
	alpha = (fstop - fstart) / pw
	theta = fstart * t + 0.5 * alpha * np.square(t)
	chirp = np.exp(1j * 2 * np.pi * theta)
	return t, chirp

################################################################################
###                            Pulse Compression                             ###
################################################################################
def _pc_numpy_convolve(sig, ref_wfm):
	"""
	Performs pulse compressing by convolving the signal (sig) with the time 
	flipped complex conjugate of the reference waveform (ref_wfm). Does this 
	using the numpy convolve function in a loop

	Parameters
	----------
	sig : ndarray of N dimensions
		Signal to be pulse compressed. Pulse compression works across the last 
		dimension
	ref_wfm : 1d ndarray
		Reference waveform for pulse compression. This function will take the 
		ref_wfm passed in, time flip it, and take the complex conjugate and 
		then convolve that with the signal to perform pulse compression. So 
		this reference waveform would be the waveform you transmitted for 
		example

	Returns
	-------
	pc : ndarray of N dimensions of type complex128
		Pulse compressed signal. Is the same number of dimensions as sig. 
		Convolution is 'valid' mode so length of last dimension is 
		max(A,B) - min(A,B) + 1 where A = sig.shape[-1] and B = ref_wfm.size
	"""
	#Compute sizes
	A = ref_wfm.size
	B = sig.shape[-1]
	pc_n = max(A,B) - min(A,B) + 1
	pc_shape = [x for x in sig.shape[:-1]]
	pc_shape.append(pc_n)
	pc = np.zeros(pc_shape, dtype=np.complex128)

	#Compute time flipped complex conjugate of reference waveform
	ref_flip_conj = np.flip(np.conjugate(ref_wfm))

	#Iterate over array and perform pulse compression across last index
	for idx in np.ndindex(sig.shape[:-1]):
		pc[idx] = np.convolve(sig[idx], ref_flip_conj, mode='valid')

	#Return pulse compressed data
	return pc

################################################################################
def _pc_scipy_convolve(sig, ref_wfm):
	"""
	Performs pulse compressing by convolving the signal (sig) with the time 
	flipped complex conjugate of the reference waveform (ref_wfm). Does this 
	using the scipy convolve function in a loop

	Parameters
	----------
	sig : ndarray of N dimensions
		Signal to be pulse compressed. Pulse compression works across the last 
		dimension
	ref_wfm : 1d ndarray
		Reference waveform for pulse compression. This function will take the 
		ref_wfm passed in, time flip it, and take the complex conjugate and 
		then convolve that with the signal to perform pulse compression. So 
		this reference waveform would be the waveform you transmitted for 
		example

	Returns
	-------
	pc : ndarray of N dimensions of type complex128
		Pulse compressed signal. Is the same number of dimensions as sig. 
		Convolution is 'valid' mode so length of last dimension is 
		max(A,B) - min(A,B) + 1 where A = sig.shape[-1] and B = ref_wfm.size
	"""
	#Compute sizes
	A = ref_wfm.size
	B = sig.shape[-1]
	pc_n = max(A,B) - min(A,B) + 1
	pc_shape = [x for x in sig.shape[:-1]]
	pc_shape.append(pc_n)
	pc = np.zeros(pc_shape, dtype=np.complex128)

	#Compute time flipped complex conjugate of reference waveform
	ref_flip_conj = np.flip(np.conjugate(ref_wfm))

	#Iterate over array and perform pulse compression across last index
	for idx in np.ndindex(sig.shape[:-1]):
		pc[idx] = signal.convolve(sig[idx], ref_flip_conj, mode='valid')

	#Return pulse compressed data
	return pc

################################################################################
def _pc_scipy_oaconvolve(sig, ref_wfm):
	"""
	Performs pulse compressing by convolving the signal (sig) with the time 
	flipped complex conjugate of the reference waveform (ref_wfm). Does this 
	using the scipy oaconvolve function in a vectorized manner

	Parameters
	----------
	sig : ndarray of N dimensions
		Signal to be pulse compressed. Pulse compression works across the last 
		dimension
	ref_wfm : 1d ndarray
		Reference waveform for pulse compression. This function will take the 
		ref_wfm passed in, time flip it, and take the complex conjugate and 
		then convolve that with the signal to perform pulse compression. So 
		this reference waveform would be the waveform you transmitted for 
		example

	Returns
	-------
	pc : ndarray of N dimensions of type complex128
		Pulse compressed signal. Is the same number of dimensions as sig. 
		Convolution is 'valid' mode so length of last dimension is 
		max(A,B) - min(A,B) + 1 where A = sig.shape[-1] and B = ref_wfm.size
	"""
	#Compute sizes
	A = ref_wfm.size
	B = sig.shape[-1]
	pc_n = max(A,B) - min(A,B) + 1
	pc_shape = [x for x in sig.shape[:-1]]
	pc_shape.append(pc_n)
	#pc = np.zeros(pc_shape, dtype=np.complex128)

	#Compute time flipped complex conjugate of reference waveform
	ref_flip_conj = np.flip(np.conjugate(ref_wfm))

	#Reshape kernel to be same dimensions as sig
	kernel_shape = [1 for x in range(len(pc_shape) - 1)]
	kernel_shape.append(ref_flip_conj.size)
	ref_flip_conj = np.reshape(ref_flip_conj, shape=kernel_shape)

	#Perform vectorized pulse compression
	pc = signal.oaconvolve(sig, ref_flip_conj, mode='valid', 
						   axes=len(kernel_shape)-1)

	#Return pulse compressed data
	return pc

################################################################################
def _pc_scipy_fftconvolve(sig, ref_wfm):
	"""
	Performs pulse compressing by convolving the signal (sig) with the time 
	flipped complex conjugate of the reference waveform (ref_wfm). Does this 
	using the scipy fftconvolve function in a vectorized manner

	Parameters
	----------
	sig : ndarray of N dimensions
		Signal to be pulse compressed. Pulse compression works across the last 
		dimension
	ref_wfm : 1d ndarray
		Reference waveform for pulse compression. This function will take the 
		ref_wfm passed in, time flip it, and take the complex conjugate and 
		then convolve that with the signal to perform pulse compression. So 
		this reference waveform would be the waveform you transmitted for 
		example

	Returns
	-------
	pc : ndarray of N dimensions of type complex128
		Pulse compressed signal. Is the same number of dimensions as sig. 
		Convolution is 'valid' mode so length of last dimension is 
		max(A,B) - min(A,B) + 1 where A = sig.shape[-1] and B = ref_wfm.size
	"""
	#Compute sizes
	A = ref_wfm.size
	B = sig.shape[-1]
	pc_n = max(A,B) - min(A,B) + 1
	pc_shape = [x for x in sig.shape[:-1]]
	pc_shape.append(pc_n)
	#pc = np.zeros(pc_shape, dtype=np.complex128)

	#Compute time flipped complex conjugate of reference waveform
	ref_flip_conj = np.flip(np.conjugate(ref_wfm))

	#Reshape kernel to be same dimensions as sig
	kernel_shape = [1 for x in range(len(pc_shape) - 1)]
	kernel_shape.append(ref_flip_conj.size)
	ref_flip_conj = np.reshape(ref_flip_conj, shape=kernel_shape)

	#Perform vectorized pulse compression
	pc = signal.fftconvolve(sig, ref_flip_conj, mode='valid', 
							axes=len(kernel_shape)-1)

	#Return pulse compressed data
	return pc

################################################################################
def compress_pulses(sig, ref_wfm):
	"""
	Performs pulse compressing by convolving the signal (sig) with the time 
	flipped complex conjugate of the reference waveform (ref_wfm). Does this 
	using the scipy fftconvolve function in a vectorized manner. I tested the 
	following methods on the same machine with a reference waveform of length 
	20e3 samples and a 2d signal of 500 rows and 200e3 columns. Compressing 
	just a single row: scipy convolve took 15 ms, scipy oaconvolve took 16 ms, 
	scipy fftconvolve took less than 1 ms, and numpy convolve took 9.6 seconds 
	meaning numpy convolve is at least 600 times slower than the others. For 
	convolving the whole matrix scipy convolve took 6.7 seconds, scipy 
	oaconvolve took 4.3 seconds, scipy fftconvolve took 4.1 seconds, and I 
	didn't even try to test numpy convolve as it would take too long. So the 
	vectorized approaches are slightly faster than the loop, which is expected. 
	And of the vectorized approaches, fftconvolve is slightly faster so we are 
	going with that as our preferred method.

	Parameters
	----------
	sig : ndarray of N dimensions
		Signal to be pulse compressed. Pulse compression works across the last 
		dimension
	ref_wfm : 1d ndarray
		Reference waveform for pulse compression. This function will take the 
		ref_wfm passed in, time flip it, and take the complex conjugate and 
		then convolve that with the signal to perform pulse compression. So 
		this reference waveform would be the waveform you transmitted for 
		example

	Returns
	-------
	pc : ndarray of N dimensions of type complex128
		Pulse compressed signal. Is the same number of dimensions as sig. 
		Convolution is 'valid' mode so length of last dimension is 
		max(A,B) - min(A,B) + 1 where A = sig.shape[-1] and B = ref_wfm.size
	"""
	return _pc_scipy_fftconvolve(sig, ref_wfm)

################################################################################
###                                Test Code                                 ###
################################################################################
if __name__ == "__main__":
	#Testing imports
	import matplotlib.pyplot as plt
	import time

	#Create chirp
	pw = 8e-3
	bw = 125e3
	fs = bw * 20
	t, chirp = gen_lfm_chirp(fs, pw, -bw/2, bw/2)

	#Plot chirp
	plt.figure()
	plt.plot(t / 1e-3, np.real(chirp), label="I")
	plt.plot(t / 1e-3, np.imag(chirp), label="Q")
	plt.xlabel("Time (ms)")
	plt.ylabel("Amplitude")
	plt.title("Chirp from 'gen_lfm'")
	plt.grid()
	plt.legend(loc="upper right")

	#Create large array of signals to speed test various pulse compression 
	#methods
	n = int(pw * 10 * fs)
	sig = np.zeros((500, n), dtype=np.complex128)
	for ii in range(sig.shape[0]):
		idx = ii * 100 + 50000
		sig[ii,idx:idx+chirp.size] = chirp

	#Plot uncompressed pulses
	plt.figure()
	plt.imshow(np.abs(sig), aspect="auto")
	plt.xlabel("Sample")
	plt.ylabel("Pulse")
	plt.title("Uncompressed Pulses")
	plt.colorbar()

	#Compress pulses
	start_time = time.monotonic()
	pc = compress_pulses(sig, chirp)
	end_time = time.monotonic()
	print("Pulse compression took %.3f seconds" % (end_time - start_time))

	#Plot compressed pulses
	plt.figure()
	plt.imshow(np.abs(pc), aspect="auto")
	plt.xlabel("Sample")
	plt.ylabel("Pulse")
	plt.title("Uncompressed Pulses")
	plt.colorbar()

	'''
	#Time various pc methods
	pc_methods = [
		("scipy convolve in loop", pc_scipy_convolve),
		("scipy oaconvolve vectorized", pc_scipy_oaconvolve),
		("scipy fftconvolve vectorized", pc_scipy_fftconvolve),
		("numpy convolve in loop", pc_numpy_convolve)
	]

	#Time a single convolution
	for pc_method in pc_methods:
		start_time = time.monotonic()
		pc_method[1](sig[0,:], chirp)
		end_time = time.monotonic()
		duration = end_time - start_time
		print("Single convolution using '%s' took %.3f seconds" % (pc_method[0], duration))

	#Time a matrix
	for pc_method in pc_methods:
		start_time = time.monotonic()
		pc_method[1](sig, chirp)
		end_time = time.monotonic()
		duration = end_time - start_time
		print("Matrix convolution using '%s' took %.3f seconds" % (pc_method[0], duration))
	'''

	#Show all plots
	plt.show()

################################################################################
###                               End of File                                ###
################################################################################