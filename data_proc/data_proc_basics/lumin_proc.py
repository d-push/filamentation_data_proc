#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:35:53 2019

@author: dmitrii

A set of functions for modes and luminescence data reading and processing.

"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import struct
import scipy.ndimage as ndimg
import os, glob
import h5py
import bottleneck as bn

from tqdm import tqdm

def rotate(data, angle):
	rot_data = ndimg.interpolation.rotate(input=data, angle=angle, order=0, prefilter=False, reshape=False)
	return rot_data

def trunc(data, left, right):
	"""
	Function truncates an array.
	"""
	trunc_data = data[:,left:right]

	return trunc_data

def read_dat(filename):
	"""
	Function opens .dat file.
	"""

	#binary data file reading
	with open(filename, "rb") as binary_file:
		data_bin = binary_file.read()

	zero = struct.unpack('>H', data_bin[0:2])[0]
	height = struct.unpack('>H', data_bin[2:4])[0]
	zero = struct.unpack('>H', data_bin[4:6])[0]
	width = struct.unpack('>H', data_bin[6:8])[0]

	try:
		s = '>H'+'H'*(height*width - 1)
		data = np.fromiter(struct.unpack(s, data_bin[8:]), dtype='uint16')
	except struct.error:
		try:
			s = '>B'+'B'*(height*width - 1)
			data = np.fromiter(struct.unpack(s, data_bin[8:]), dtype='uint8')
			#data = data*8
			#print("Warning: 8bit image. Read with adjustment to 12bit format (magnified by 8).")
		except:
			print("ERROR: could not read data file {}".format(filename))
			return None

	data = np.reshape(data, (height, width))
	if width < height:
		return(data.T, height, width) #Transpose array and swap height and width.
	else:
		return(data, width, height)

def read_raw_Mind_Vision(filename, dims=(1280,960), writing_gain=16):
	"""
	Function opens .RAW file from chineese CCD. The data should be in format of 16-bit raw matrix.

	Parameters:
	-------------------------------------------
	- filename : path (string)
		Path to the file to read data from.
	- writing_gain:
		Коэффициент, на которые нужно разделить данные для получения исходных значений.
	"""

	#Constants.
	width = dims[0] #Длина кадра.
	height = dims[1] #Ширина кадра.

	#binary data file reading
	with open(filename, "rb") as binary_file:
		data_bin = binary_file.read()

	f_size = width*height

	try:
		s = '<H'+'H'*(f_size - 1)
		data = np.fromiter(struct.unpack(s, data_bin), dtype='uint16')
	except struct.error:
		print("In data_proc/lumin_proc, in function 'read_raw_Mind_Vision':")
		print("ERROR: could not read data file {}".format(filename))
		return None

	data = np.reshape(data, (height, width))
	data = data/writing_gain #Devide by 'gain' introduced by writing of the 12-bit image into 16 bit.

	return(data, width, height)

def read_modes_basing_on_ext(filename_mode, ext, mult=False, dims=(1280,960), writing_gain=16):
	'''
	Function for read mode in dependence of file extension.
	'''
	if ext=='.RAW':
		try:
			data, width, height = read_raw_Mind_Vision(filename_mode, dims=dims, writing_gain=writing_gain)
		except (TypeError):
			print("Error while reading file {}.".format(filename_mode))
			return(None)
	elif ext=='.dat':
		#Обработка исключения в случае, если не удаётся прочитать .dat файл.
		try:
			data, width, height = read_dat(filename_mode)
		except (TypeError, struct.error):
			print("Error while reading file {}.".format(filename_mode))
			return(None)
	elif ext=='.txt':
		try:
			data = np.loadtxt(filename_mode)
			height, width = data.shape
		except ValueError:
			try:
				data = np.loadtxt(filename_mode, skiprows=1)
				data = data[:,1:]
				height, width = data.shape
			except ValueError:
				print("Error while reading file {}.".format(filename_mode))
				return(None)
	elif ext=='.csv':
		with open(filename_mode, 'r') as f:
			lines = f.readlines()
			try:
				data = np.loadtxt(line.replace(';',' ') for line in lines)
				height, width = data.shape
			except ValueError:
				try:
					data = np.loadtxt(line.replace(';',' ') for line in lines[1:])
					data = data[:,1:]
					height, width = data.shape
				except ValueError:
					print("Error while reading file {}.".format(filename_mode))
					return(None)
	elif ext=='.npy':
		data=np.load(filename_mode)
		height, width = data.shape
	elif ext=='.png' or ext=='.tif':
		try:
			data = plt.imread(filename_mode)
			height, width = data.shape
		except OSError:
			print("Error while reading file {}.".format(filename_mode))
			return(None)
	elif ext=='.bgData':
		if not mult:
			with h5py.File(filename_mode, 'r') as f:
				data = f['BG_DATA']['1']['DATA'][()]
				height = f['BG_DATA']['1']['RAWFRAME']['HEIGHT'][()][0]
				width = f['BG_DATA']['1']['RAWFRAME']['WIDTH'][()][0]
				data = np.reshape(data, (height, width))
				data = data/2**19
		else:
			mult_data = []
			with h5py.File(filename_mode, 'r') as f:
				data_list = f['BG_DATA']
				for i in range(1, len(data_list)+1):
					data = data_list[str(int(i))]['DATA'][()]
					height = data_list[str(int(i))]['RAWFRAME']['HEIGHT'][()][0]
					width = data_list[str(int(i))]['RAWFRAME']['WIDTH'][()][0]
					data = np.reshape(data, (height, width))
					data = data/2**19
					mult_data.append(data)
			return(mult_data, width, height)
	else:
		print("In data_proc, in file lumin_proc, in function read_modes_basing_on_ext:")
		print(f'ERROR: unknown extension {ext}')
		return(None)
	return(data, width, height)

def read_dat_float(filename):
	"""
	Function opens .dat file.
	"""

	#binary data file reading
	with open(filename, "rb") as binary_file:
		data_bin = binary_file.read()

	zero = struct.unpack('>H', data_bin[0:2])[0]
	height = struct.unpack('>H', data_bin[2:4])[0]
	zero = struct.unpack('>H', data_bin[4:6])[0]
	width = struct.unpack('>H', data_bin[6:8])[0]

	try:
		s = '>d'+'d'*(height*width - 1)
		data = np.fromiter(struct.unpack(s, data_bin[8:]), dtype='float64')
	except struct.error:
		try:
			s = '>B'+'B'*(height*width - 1)
			data = np.fromiter(struct.unpack(s, data_bin[8:]), dtype='uint8')
			data = data*8
			print("Warning: 8bit image. Read with adjustment to 12bit format (magnified by 8).")
		except:
			print("ERROR: could not read data file {}".format(filename))
	data = np.reshape(data, (height, width))

	return(data, width, height)

def save_dat(filename_to_save, data):
	'''
	Write 2d array to .dat file.
	'''
	fout = open(filename_to_save, 'wb')
	fout.write(struct.pack('>H', 0))
	fout.write(struct.pack('>H', data.shape[0]))
	fout.write(struct.pack('>H', 0))
	fout.write(struct.pack('>H', data.shape[1]))
	s = '>H'+'H'*(data.size - 1)
	data = data.flatten()
	fout.write(struct.pack(s, *data))
	fout.close()

def save_dat_float(filename_to_save, data):
	'''
	Write 2d array to .dat file.
	'''
	fout = open(filename_to_save, 'wb')
	fout.write(struct.pack('>H', 0))
	fout.write(struct.pack('>H', data.shape[0]))
	fout.write(struct.pack('>H', 0))
	fout.write(struct.pack('>H', data.shape[1]))
	s = '>d'+'d'*(data.size - 1)
	data = data.flatten()
	fout.write(struct.pack(s, *data))
	fout.close()

def read_bd_map(bd_map_file):
	"""
	Read breakdown map, as specified in bd_map_file.
	It is assumed that coordinates of breakdowns are sorted by x increasing.
	"""
	#Variables
	bd_mult = [] #Координаты "множественных" пробоев.
	bd_single = [] #Координаты "одиночных" пробоев.

	f=open(bd_map_file)
	lines = f.readlines()
	print("Reading breakdown map...")
	if lines[0] == "Muliple hot spots\n":
		print("File start is OK.")
	single_start_num = lines.index("Separate hot spots\n")
	bd_mult = np.genfromtxt(bd_map_file, skip_header=1, max_rows=single_start_num-1, dtype = 'uint16')
	bd_single = np.genfromtxt(bd_map_file, skip_header=single_start_num+1, dtype = 'uint16')
	if len(bd_single.shape) == 1:
		bd_single.reshape(bd_single.shape[0],-1)
	print("Bd map has been successfully read.")

	return (bd_mult, bd_single)

def apply_bd_map(data, bd_mult, bd_single):
	"""
	Removes breakdowns, which coordinates are listed in bd_mult ("multiple" breakdowns) and bd_single (single separated hot spots).
	It is assumed that coordinates of breakdowns are sorted by x increasing.
	"""
	#Для множественных пробоев - аппроксимируем пробитый участок линейной зависимостью, исходя из ближайших непробитых точек.
	i_old = 0
	if bd_mult.size != 0:
		for k in range(bd_mult.shape[0]):
			j,i = bd_mult[k]
			if i == i_old and j>j_bottum and j<j_top:
				continue
			elif i < i_old:
				print("ERROR: coordinates are not sorted by i increasing!!!")
				return "apply_bd_map_error"
			else:
				i_old = i; k1 = k
				while k1<bd_mult.shape[0] and bd_mult[k1,1] == i: #Используем, что массив отсортирован по i, а где i одинаково - по j.
					k1 += 1
				j_top = int(bd_mult[k1-1,0])+1; k1=k #Ближайшая непробитая точка сверху.
				while k1>=0 and bd_mult[k1,1] == i:
					k1 -= 1
				j_bottum = int(bd_mult[k1+1,0])-1 #Ближайшая непробитая точка снизу.
				# Коэффициенты линейной аппроксимации.
				if j_top >= bd_mult.shape[0]: #Пробой на верхнем краю ПЗС.
					data[j_bottum:j_top,i] = np.around(np.ones(j_top - j_bottum)*np.mean(data[j_bottum-1:j_bottum-4,i]))
				elif j_bottum <= 0:
					data[j_bottum+1:j_top+1,i] = np.around(np.ones(j_top - j_bottum)*np.mean(data[j_top+1:j_top+4,i]))
				else:
					A = (float(data[j_top,i]) - float(data[j_bottum,i]))/(float(j_top) - float(j_bottum))
					B = float(data[j_bottum,i]) - A*j_bottum
					data[j_bottum+1:j_top,i] = np.around(A*np.arange(j_bottum+1, j_top)+B)

	#Для одиночных пробоев.
	if bd_single.size != 0:
		for k in range(bd_single.shape[0]):
			j,i = bd_single[k]
			vic_sum = np.sum(data[j-1:j+2,i-1:i+2]) - data[j,i]
			data[j,i] = vic_sum/8.0
	return data

def remove_hot_spots(data, threshold = 0.5, window=3):
	"""
	Looks for single pixel hot spots and removes them from the data array.
	Window should be even.
	"""
	bord_width = int(np.floor((window-1)/2.0))
	data1 = np.zeros_like(data[window-1:,window-1:])
	count = 0
	for i in range(bord_width, data.shape[0]-bord_width-1):
		for j in range(bord_width, data.shape[1]-bord_width-1):
			border_sum = np.sum(data[i-1:i+2,j-1:j+2]) - data[i,j]
			border_mean = border_sum/(window**2 - 1)
			if data[i,j] == 0:
				data1[i,j] = data[i,j]
			else:
				if border_mean < data[i,j]*threshold:
					data1[i,j] = border_mean
					print(i,j)
					count+=1
				else:
					data1[i,j] = data[i,j]
	print(f'count = {count}')

	return(data1)
	
def remove_hot_spots1(data, threshold = 0.5, window=3):
	"""
	Looks for single pixel hot spots and removes them from the data array.
	Window should be even.
	"""
	bord_width = int(np.floor((window-1)/2.0))
	data1 = ndimg.median_filter(data, size=window)
	mask = np.argwhere(data1 < threshold*data)
	data2 = np.copy(data)
	count = 0
	for coords in mask:
		print(coords)
		if (coords[0] < bord_width) or (coords[1] < bord_width) or (coords[0] > data.shape[0]-bord_width-1) or (coords[1] > data.shape[1]-bord_width-1):
			data2[coords[0], coords[1]] = data1[coords[0], coords[1]]
		else: 
			border_sum = np.sum(data[coords[0]-1:coords[0]+2,coords[1]-1:coords[1]+2]) - data[coords[0],coords[1]]
			border_mean = border_sum/(window**2 - 1)
			data2[coords[0], coords[1]] = border_mean
	print(f'count = {count}')

	return(data2)
	
def run_av_2d(data, window=11, axis=1):
	'''
	Function performs running average on 2d data array along x axis.
	window should be even.
	'''
	data_1 = np.zeros_like(data[:,window-1:])
	for i in range(0, data.shape[axis]-(window-1)):
		data_1[:,i] = np.mean(data[:,i:i+window], axis = axis)

	return(data_1)

def run_av(data, window=11):
	'''
	Function performs running average on 1d data array.
	window should be even.
	'''
	data_1 = bn.move_mean(data, window=window)[window-1:]
	#data_1 = np.zeros_like(data[window-1:])
	#for i in range(0, data.shape[0]-(window-1)):
	#	data_1[i] = np.mean(data[i:i+window])

	return(data_1)

def find_limits(data, method='simple', add_sigma = None):
	'''
	Search limits (v_min and V_max) for 2D array plotting.

	Parameters:
	----------------------
	data : 2D array
		Data array limits search
	method : 'simple' or 'good'
		Max search algorithm: 'simple' - to pick up an absolute maximum of the array, 'good' - to use maximum average of 9 pixel square.
	add_sigma : float
		A number of sigma added for background value for v_min. None - use v_min as is. v_min is searched as average of four 20x20-pixel squares at the array angles.

	'''
	#Константы
	width = 3
	threshold_multiplier = 1.5
	if method=='simple':
		v_max = np.amax(data)
	elif method=='good':
		#Максимальное среднее по квадрату 3x3.
		v_max = np.amax(ndimg.median_filter(data, size=width))
	elif method=='clever':
		i,j = np.argmax(data)
		if data[i-1:i+2,j-1:j+2] > threshold_multiplier*data[i-1:i+2,j-1:j+2]:
			v_max = np.amax(ndimg.median_filter(data, size=width))
		else:
			v_max = np.amax(data)
	else:
		print("ERROR: in find_limits (lumin_proc.py) - unknown keyword for scale maxima search")
		return False
	v_min = (np.sum(data[:20, :20]) + np.sum(data[-20:, :20]) + np.sum(data[:20,-20:]) + np.sum(data[-20:,-20:]))/1600.0

	if add_sigma:
		angle_squares = np.hstack((data[:20, :20], data[-20:, :20], data[:20,-20:], data[-20:,-20:]))
		sigma = np.std(angle_squares)
		v_min = v_min + add_sigma*sigma

	return (v_min, v_max)

def subtract_plane(data, quite=False):
	'''
	Function subtracts an inclined plane from data background.
	'''

	#Constants.
	stripe_width = 5 #Ширина полосы по краям кадра (в пикселях), используемая для расчёта параметров вычитаемой плоскости.

	#Выделяем полосы вдоль сторон массива шириной stripe_width без повторения элементов.
	data1 = data[:stripe_width,].flatten()
	data2 = data[stripe_width:-stripe_width, :stripe_width].flatten()
	data3 = data[stripe_width:-stripe_width, -stripe_width:].flatten()
	data4 = data[-stripe_width: ,].flatten()

	values = np.hstack((data1, data2, data3, data4)) #Cоединяем значения выбранных элементов в один одномерный массив.

	#Готовим массив X ординат (номеров) элементов из массива values.
	x1 = np.tile(np.arange(0, data.shape[1]), stripe_width)
	x2 = np.tile(np.arange(0, stripe_width), data.shape[0]-2*stripe_width)
	x3 = np.tile(np.arange(data.shape[1]-stripe_width, data.shape[1]), data.shape[0]-2*stripe_width)
	x4 = np.tile(np.arange(0, data.shape[1]), stripe_width)

	X = np.hstack((x1, x2, x3, x4))

	#Готовим массив Y ординат (номеров) элементов из массива values.
	y1 = np.repeat(np.arange(0, stripe_width), data.shape[1])
	y2 = np.repeat(np.arange(stripe_width, data.shape[0]-stripe_width), stripe_width)
	y3 = np.repeat(np.arange(stripe_width, data.shape[0]-stripe_width), stripe_width)
	y4 = np.tile(np.arange(data.shape[0]-stripe_width, data.shape[0]), data.shape[1])

	Y = np.hstack((y1, y2, y3, y4))

	Z = np.ones_like(X)
	coords = np.vstack((X,Y,Z)).T #Массив с "правильной" (с т.зр. перемножения матриц) размерностью.

	A, B, C = np.linalg.lstsq(coords, values, rcond=None)[0] #МНК
	if not quite:
		print(f'Subtracted plane coefficients: A = {A}, B = {B}, C = {C}')

	I = np.arange(0, data.shape[1])
	J = np.arange(0, data.shape[0])
	ii, jj = np.meshgrid(I,J, indexing='xy')

	plane_data = A*ii+B*jj+C
	data = data - plane_data

	return(data)

def subtract_min_corner(data):
	'''
	Function subtracts constant from the data, basing on the minimum average value of data array corners.
	'''
	
	#Constants.
	corner_size = 20 #px
	
	#Выделяем полосы вдоль сторон массива шириной stripe_width без повторения элементов.
	corner1 = np.mean(data[:corner_size,:corner_size])
	corner2 = np.mean(data[-corner_size-1:,:corner_size])
	corner3 = np.mean(data[:corner_size,-corner_size-1:])
	corner4 = np.mean(data[-corner_size-1:,-corner_size-1:])

	bg_value = np.amin((corner1, corner2, corner3, corner4))
	
	data = data - bg_value
	
	return(data)

def find_centre_cycle(data, x, y, gr_len_x, gr_len_y, eps):
	'''
	Cycle for find_centre function.
	!!! Function CHANGES input array (data). !!!
	'''

	x_old = -10; y_old = -10 #Temporal values; should not concide with initial x and y values.

	while (abs(x - x_old) > eps) or (abs(y - y_old) > eps):
		x_old = x
		y_old = y
		x = 0.0
		y = 0.0
		s = 0.0
		if int(x_old)-gr_len_x >= 0:
			j_min = int(x_old)-gr_len_x
		else:
			j_min = 0
		if int(x_old)+gr_len_x <= data.shape[1]-1:
			j_max = int(x_old)+gr_len_x
		else:
			j_max = data.shape[1]-1

		if int(y_old)-gr_len_y >= 0:
			i_min = int(y_old)-gr_len_y
		else:
			i_min = 0
		if int(y_old)+gr_len_y <= data.shape[0]-1:
			i_max = int(y_old)+gr_len_y
		else:
			i_max = data.shape[0]-1

		#%% Calculate mass centre coordinates.
		i = np.arange(i_min, i_max)
		j = np.arange(j_min, j_max)
		x = np.sum(j*np.sum(data[i_min:i_max,j_min:j_max], axis=0)) #y mass centre
		y = np.sum(i*np.sum(data[i_min:i_max,j_min:j_max], axis=1)) #x mass centre. Axis = number of the axis, along which the sum is calculated.
		s = np.sum(data[i_min:i_max,j_min:j_max])

		x = x/s
		y = y/s

	return x, y


def find_centre(input_array, xc_init, yc_init, crop_width, crop_height, eps=1.0, n_fon=20):
	'''
	Assess mass centre of the array. Returns centre coordinates as float values.
	x correspond to the j (column) number of the array!
	'''

	#Constants
	filt_sigma = 4 #radius for gaussian filter.
	threshold = 0.15 #from data_max. Points below this value would not be considered.

	data = np.copy(input_array) #Make an independent copy of the initial array.

	x = xc_init; y = yc_init #Initial assumptions for x_centre and y_centre.
	#x_old = -10; y_old = -10 #Temporal values; should not concide with initial x and y values.

	gr_len_x = int(round(crop_width/2.0)) #Half-width of the region to be cropped out.
	gr_len_y = int(round(crop_height/2.0)) #Half-height of the region to be cropped out.

	####TEMPORALY_REMOVED####
	#max_fon1 = np.amax(data[:n_fon,:n_fon])
	#max_fon2 = np.amax(data[-n_fon:,:n_fon])
	#max_fon3 = np.amax(data[-n_fon:,:n_fon])
	#max_fon4 = np.amax(data[-n_fon:,-n_fon:])

	#max_fon = np.amax((max_fon1, max_fon2, max_fon3, max_fon4))

	#if np.amax(data) > max_fon:
	#	data = data-max_fon
	#	data[data < 0] = 0

	####NEW_METHOD####
	data = ndimg.gaussian_filter(data, filt_sigma)
	data_max = np.amax(data)
	data = data + (1.0 - data_max*threshold)
	data[data < 1.0] = 1.0
	data = np.log10(data)
	###################
	
	####OLD_Method####
	#data_min = np.amin(data[data > 0])
	#data[data<=0] = data_min
	#data = np.log10(data) + np.abs(np.log10(data_min))
	#data_max = np.amax(data)
	#fon = np.hstack((data[:n_fon,:n_fon], data[-n_fon:,:n_fon], data[-n_fon:,:n_fon], data[-n_fon:,-n_fon:]))
	#fon_std = np.std(fon)
	#data[data<(data_max-fon_std)/2.0] = 0
	##################

	x,y = find_centre_cycle(data, x, y, gr_len_x, gr_len_y, eps)

	return(int(round(x)),int(round(y)))

def calc_bg_from_empty_frames(filenames_modes, ext, bd_mult, bd_single, n_fon=20, subtr_plane=True):
	'''
	Calculate average background based on 'empty' frames.
	'''

	#Constants
	filt_size = 5 #Size for uniform filter.
	n_sigma = 5 #File is "empty" when filtered array maximum < filtered bg maximum multiplied by this number.

	data = read_modes_basing_on_ext(filenames_modes[0], ext)[0]

	data_bg = np.zeros_like(data, dtype=float)
	bg_count = 0
	empty_filenames = []

	print("\nLooking for 'empty' frames...")
	with tqdm(total = len(filenames_modes)) as psbar:
		for i, filename_mode in enumerate(filenames_modes):

			#%%Читаем файл в зависимости от разширения.
			data = read_modes_basing_on_ext(filename_mode, ext)
			if data is not None:
				data = data[0]
			else:
				continue

			#%%Учитываем пробои.
			if (len(bd_mult) > 0) and (len(bd_single) > 0):
				data = apply_bd_map(data, bd_mult, bd_single)

			#%%Вычитаем плоскость
			if subtr_plane:
				data = subtract_plane(data, quite=True)

			#%%Считаем стандартное отклонение фона и данных.
			fon_arr = np.array((data[:n_fon,:n_fon], data[-n_fon:,:n_fon], data[:n_fon,-n_fon:], data[-n_fon:,-n_fon:]))
			for i,arr in enumerate(fon_arr):
				fon_arr[i] = ndimg.median_filter(arr, size=filt_size)

			fon_mean = np.mean(fon_arr)
			fon_std = np.amax(np.std(fon_arr, axis = (1,2)))
			fon_max = fon_mean + fon_std*n_sigma

			frame_filt = ndimg.median_filter(data, size=filt_size)
			frame_filt_max = np.amax(frame_filt)

			if frame_filt_max <= fon_max:
				data_bg += data
				bg_count += 1
				empty_filenames.append(filename_mode)
			#if frame_std <= fon_std:
			#	data_bg += data
			#	bg_count += 1
			#	empty_filenames.append(filename_mode)

			psbar.update(1) #Обновляем строку состояния.

	if bg_count > 0:
		print(f"{bg_count} 'empty' files have been found.")
		print("Empty filenames:")
		print(empty_filenames)
		data_bg = data_bg/bg_count
		return(data_bg, bg_count)
	else:
		print("No 'empty' files")
		return(None)

def create_default_empty_frame(foldernames_modes, ext, bd_mult, bd_single, n_fon=20, subtr_plane=True):


	print("\nCreating default background file. It will be use for data sets, where no 'empty' frames are detected.")
	file_mask = '*_*'+ext
	for foldername in foldernames_modes:
		print(f'Search using mask: {os.path.join(foldername, file_mask)}')
		filenames_modes = glob.glob(os.path.join(foldername, file_mask))
		empty = calc_bg_from_empty_frames(filenames_modes, ext, bd_mult, bd_single, n_fon=20, subtr_plane=True)

		if empty is not None:
			print("Default background array from empty files has been cretated.")
			return(empty)
	return(None)
