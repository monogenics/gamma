from math import pi, sqrt, atan
from random import random, randint, choice
import time
import ast
import csv
import numpy as np
import matplotlib.pyplot as plt
import cProfile


def LengthToGamma(lengths):

	lengths=(np.asarray(lengths)*10)

	wavelength = 637
	#wavelength = 420
	n_eff = 1.9790286464722768

	# N = 20
	# z = 20e3

	# N = 15
	# z = 10e3
	# NA = .150 #Numerical Aperture

	# N = 8
	z = 2.5e3
	NA = .25 #Numerical Aperture

	w = wavelength / (pi * NA) #mode field diameter

	k = 2*pi/wavelength

	zr = pi*w*w/wavelength

	W = w*sqrt(1 + (z/zr)**2)
	R = z*(1 + (zr/z)**2)
	phi = atan(z/zr)

	debug = 0

	r00 = complex(0.03830434239995499, 0.1365633292253801)
	t00 = complex(0.5607483104194094, -0.7632956574679816)
	su0 = 0.1997214854358504
	sd0 = 0.18093412141177265

	# MATRICES #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

	def scatter_matrix(width):

		return np.matrix([  [ 1/t00,     np.conj(r00)/t00             ],
							[ r00/t00,   (r00*np.conj(r00) + t00*np.conj(t00))/t00   ]  ],
							dtype=complex)

	def propagate_matrix(length):
		return np.matrix([  [ np.exp(pi*2j*n_eff*length/wavelength),              0                     ],
							[ 0,                                 np.exp(-pi*2j*n_eff*length/wavelength)   ]  ],
							dtype=complex)

		# return np.matrix([  [np.exp(          pi * 2j * n_eff_fund  * length / wavelength), 0, 0, 0],
		#                     [0, np.exp(      -pi * 2j * n_eff_fund  * length / wavelength),    0, 0],
		#                     [0, 0, np.exp(    pi * 2j * n_eff_first * length / wavelength),       0],
		#                     [0, 0, 0, np.exp(-pi * 2j * n_eff_first * length / wavelength)         ]  ],
		#                     dtype=complex)

	# AMPLITUDES #### #### #### #### #### #### #### #### #### ####
	def getAmplitudes(widths, lengths):
		N = widths.shape[0]                                 # N grates.

		a_fund = np.zeros((2, 2*N), dtype=complex)                              # N times [[ a  b  ]; [ b' a' ]].

		a_fund[0, 2*N-1] = 1                                     # Set the initial vector (b = 1, a' = 0).
		a_fund[:, 2*N-2] = a_fund[:, 2*N-1] * scatter_matrix(widths[N-1])   # And find a, b' for the first grate.

		# print(scatter_matrix(widths[0]))

		for ii in range(N-2, -1, -1):                               # Now do this for the rest of the grates.
			a_fund[:, 2*ii+1] = propagate_matrix(lengths[ii]).dot(a_fund[:, 2*ii+2])
			a_fund[:, 2*ii] =   scatter_matrix(widths[ii])   .dot(a_fund[:, 2*ii+1])

		# a_first = np.zeros((4, 2*N), dtype=complex)                              # N times [[ a  b  ]; [ b' a' ]].
		#
		# a_first[2, 2*N-1] = 1                                     # Set the initial vector (b = 1, a' = 0).
		# a_first[:, 2*N-2] = a_first[:, 2*N-1] * scatter_matrix(widths[N-1])   # And find a, b' for the first grate.
		#
		# for ii in range(N-2, -1, -1):                               # Now do this for the rest of the grates.
		#     # a_first[:, 2*ii+1] = a_first[:, 2*ii+2] * propagate_matrix(lengths[ii])
		#     # a_first[:, 2*ii] =   a_first[:, 2*ii+1]  * scatter_matrix(widths[ii])
		#     a_first[:, 2*ii+1] = propagate_matrix(lengths[ii]).dot(a_first[:, 2*ii+2])
		#     a_first[:, 2*ii] =   scatter_matrix(widths[ii])   .dot(a_first[:, 2*ii+1])
		#
		# a = a_fund - (a_fund[2,0]/a_first[2,0]) * a_first

		# print(np.abs(a_fund[2,0]/a_first[2,0]))

		return a_fund


	# OVERLAP #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
	# def E(x,w,z):
	#     k = 2*pi/wavelength
	#
	#     zr = pi*w*w/wavelength
	#
	#     W = w*sqrt(1 + (z/zr)**2)
	#     R = z*(1 + (zr/z)**2)
	#     phi = atan(z/zr)
	#
	#     return np.exp(-x**2 / W**2) * np.exp(-1j *(k*z + k*x*x/(2*R) - phi) )

	def E(x,w,z):
		# return np.exp(-x**2 / W**2) * np.exp(-1j * (k*z + k*x*x/(2*R) - phi) )
		return np.exp(-x**2 / W**2) * np.exp(1j * k*x*x/(2*R) )
		# d = (2 * pi / wavelength) * (400/800);
		# return d if x == 0 else ( d * np.sin(x * d) / (pi * x * d)) * np.exp(-1j *(k*z + k*x*x/(2*R) - phi) )
		# return ( 50000 * np.sin(x / d) / (pi * x / d)) * np.exp(-1j *(k*z + k*x*x/(2*R) - phi) )

	def getOverlap(widths, lengths, amplitudes, W, Z, num_notches, outputScatter):
		other_notches = 5;

		tot_notches = num_notches + 2*other_notches

		other_notch_space = 200

		s =     np.zeros(tot_notches, dtype = complex)
		sd =    np.zeros(tot_notches, dtype = complex)
		x =     np.zeros(tot_notches)
		dx =    np.ones(tot_notches)*other_notch_space

		#for i in range(0, num_notches):
			#print("Scatter:", scatter(widths[i]))
			#print("Amplitudes:", amplitudes[0,2*i] + amplitudes[1,2*i+1])

		currentx = -widths[0]/2;

		for i in range(0, num_notches):
			s[i+other_notches] =  su0*(amplitudes[0,2*i] + amplitudes[1,2*i+1]) #+ scatter_dict["su1"]*(amplitudes[2,2*i] + amplitudes[3,2*i+1])
			sd[i+other_notches] = sd0*(amplitudes[0,2*i] + amplitudes[1,2*i+1]) #+ scatter_dict["sd1"]*(amplitudes[2,2*i] + amplitudes[3,2*i+1])
			x[i+other_notches] = currentx + widths[i]/2;

			if 		i == 0:
				dx[i+other_notches] = widths[i] + lengths[i]
			elif 	i == num_notches-1:
				dx[i+other_notches] = widths[i] + lengths[i-1]
			else:
				dx[i+other_notches] = widths[i] + (lengths[i-1] + lengths[i])/2

			if 	i != num_notches-1:
				currentx += lengths[i] + widths[i];

		for i in range(0, other_notches):
			x[other_notches-i-1] = x[other_notches-i] - other_notch_space - 100;
			x[other_notches+num_notches+i] = x[other_notches+num_notches-1+i] + other_notch_space + 100;

		s  /= amplitudes[0,0]
		sd /= amplitudes[0,0]

		# if debug:
		#     S = np.zeros(num_notches)
		#
		#     for i in range(0, num_notches):
		#         S[i] = 100 * np.abs(s[i])**2
		#
		#     print(S)


		scatter_up = 0
		scatter_down = 0

		for i in range(0, tot_notches):
			scatter_up      += np.abs( s[i])**2
			scatter_down    += np.abs(sd[i])**2

		# np.set_printoptions(suppress=True)
		# np.set_printoptions(precision=3)
		# print(np.abs(amplitudes/amplitudes[0,0])**2)
		# print(np.angle(amplitudes/amplitudes[0,0]))

		final_reflection =          np.abs(amplitudes[1,0]                  /amplitudes[0,0])**2
		final_transmission =        np.abs(amplitudes[0,2*num_notches-1]    /amplitudes[0,0])**2
		# final_reflection_first =    np.abs(amplitudes[3,0]                  /amplitudes[0,0])**2
		# final_transmission_first =  np.abs(amplitudes[2,2*num_notches-1]    /amplitudes[0,0])**2
		final_reflection_first =    0
		final_transmission_first =  0

		final_gamma = 0
		final_X = 0;
		final_scatter_up = 0
		final_scatter_down = 0
		sensitivity = 100;
		Wround = int(min(currentx, W)/200)*100;
		# print(range(Wround, int(np.max(x)) + sensitivity-Wround, sensitivity));

		gammav2 = True

		'''
		plt.axis([0, 7000, 0.0, 20.0])
		plt.xlabel("Position (nm)")
		plt.ylabel("scatter (%)")
		plt.ion()
		plt.show()
		'''

		for X in range(Wround, int(currentx) + sensitivity-Wround, sensitivity):
			gamma = 0
			integral = 0

			for i in range(0, tot_notches):
				if gammav2:
					# gamma           += s[i] * np.conj(E(x[i] - X, W, Z)) * dx[i]
					# integral        += (np.abs(E(x[i] - X, W, Z))**2) * (dx[i]**2)
					gamma           += s[i] * np.conj(E(x[i] - X, W, Z)) * sqrt(dx[i])
					integral        += (np.abs(E(x[i] - X, W, Z))**2) * dx[i]
				else:
					gamma           += s[i] * np.conj(E(x[i] - X, W, Z))
					integral        += (np.abs(E(x[i] - X, W, Z))**2)

			gamma = (np.abs(gamma/integral)**2)
			# gamma = (np.abs(gamma)**2)

			# if outputScatter:
			# 	print("{:.2f}\t{:.2f}\t{:.2f}".format(X, 100*gamma, 100*integral))

			if gamma > final_gamma:
				final_gamma = gamma
				final_X = X;

		if outputScatter:
			#print(range(Wround, int(currentx) + sensitivity-Wround, sensitivity))

			gamma = 0
			integral = 0

			for i in range(0, tot_notches):
				if gammav2:
					# gamma           += s[i] * np.conj(E(x[i] - X, W, Z)) * dx[i]
					# integral        += (np.abs(E(x[i] - X, W, Z))**2) * (dx[i]**2)
					gamma           += s[i] * np.conj(E(x[i] - X, W, Z)) * sqrt(dx[i])
					integral        += (np.abs(E(x[i] - X, W, Z))**2) * dx[i]
				else:
					gamma           += s[i] * np.conj(E(x[i] - X, W, Z))
					integral        += (np.abs(E(x[i] - X, W, Z))**2)

			# for i in range(0, tot_notches):
			# 	# print("{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}".format(x[i], 100*(np.abs(s[i])**2)/dx[i], np.angle(s[i]), 100*(np.abs(E(x[i] - final_X, W))**2)/dx[i]/integral))
			# 	# print("{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}".format(x[i], 100*(np.abs(s[i])**2) / dx[i], np.angle(s[i]), 100*(np.abs(E(x[i] - final_X, W))**2)*dx[i]/integral))
			# 	print("{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}".format(x[i], 100*(np.abs(s[i])**2), np.angle(s[i]), 100*(np.abs(E(x[i] - final_X, W, Z))**2)/integral))

			# print(np.angle(np.transpose(amplitudes)/amplitudes[0,0]));

			# plt.plot(x, 100*(np.abs(s)**2)/dx,'bo-',label='Scatter')
			# plt.plot(x, 1*(np.angle(s) + pi),'ro-',label='Phase')
			# plt.plot(x, 100*(np.abs(E(x - final_X, W))**2)/dx/integral,'go-',label='Match')
			'''
			plt.gcf().clear()
			plt.plot(x, 500*(np.abs(s)**2),'bo-',label='Scatter')
			plt.plot(x, 5000*(np.abs(s)**2)/sqrt(dx[i]),'co-',label='Scatter')
			plt.plot(x, 1*(np.angle(s) + pi),'ro-',label='Phase')
			plt.plot(x, 10*(np.abs(E(x - final_X, W, Z))**2),'go-',label='Match')
			plt.plot(x, 1*(np.angle(E(x - final_X, W, Z)) + pi),'ko-',label='Match')
			plt.plot(x, dx/100,'yo-',label='Match')
			plt.draw()
			plt.pause(.00001)
			'''

			# print [X, gamma]
		# print final_X;

		# total_scatter_up =      np.sum(np.abs(s)**2);
		# total_scatter_down =    np.sum(np.abs(sd)**2);

		return [final_gamma,
				final_transmission,
				final_reflection,
				final_transmission_first,
				final_reflection_first,
				final_X, scatter_up,
				scatter_down]
				#, total_scatter_up, total_scatter_down]

	def main(lengths):
		N = len(lengths)+1;
		widths = 100 * np.ones(N)
		amplitudes = getAmplitudes(widths, lengths)
		#print(getOverlap(widths, lengths, amplitudes, w, z, N, False)[0])
		return getOverlap(widths, lengths, amplitudes, w, z, N, True)[0]

	return main(lengths)

def main():
	#TESTING: Note that every grating that is tested must have at least seven (7) notches.
	lengths_list = [[143, 313, 328, 135, 132, 167, 165],
					[181, 152, 307, 100, 259, 100, 199],
					[177, 145, 113, 100, 259, 100, 100],
					[174, 145, 316, 100, 267, 100, 389],
					[160, 340, 299, 95, 281, 299, 280],
					250 * np.ones(20),
					260 * np.ones(20),
					270 * np.ones(20),
					280 * np.ones(20),
					290 * np.ones(20)]
	for lengths in lengths_list:
		print(LengthToGamma(lengths))

if __name__ == '__main__':
	main()
