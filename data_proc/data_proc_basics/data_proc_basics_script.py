#from __future__ import division
import os, sys
import numpy as np
import struct

from .lumin_proc import *

#%%Чтение бинарных файлов с осциллографа.
def read_bin(filepath):
	with open(filepath, "rb") as binary_file:
		# Read the whole file at once
		data = binary_file.read()

	T = struct.unpack('>l', data[0:4])[0]
	E = struct.unpack('>d', data[4:12])[0]
	dt = struct.unpack('>d', data[12:20])[0]
	dV = struct.unpack('>d', data[20:28])[0]

	wf_len = len(data)-28
	s = '>b'+ 'b'*(wf_len-1)

	waveform = np.fromiter(struct.unpack(s, data[28:]), dtype='int8')
	waveform = waveform*dV

	print("filename = {4}, T = {0}, E = {1}, dt = {2}, dV = {3}".format(T, E, dt, dV, filepath))

	return(dt, dV, waveform)

#%%Чтение бинарных файлов с осциллографа.
def read_bin_new_program(filepath):
	with open(filepath, "rb") as binary_file:
		# Read the whole file at once
		data = binary_file.read()

	E = struct.unpack('>l', data[0:4])[0]
	T = (struct.unpack('15s', data[4:19])[0]).decode('UTF-8')
	dt = struct.unpack('>d', data[19:27])[0]
	dV = struct.unpack('>d', data[27:35])[0]

	wf_len = len(data)-35
	s = '>b'+ 'b'*(wf_len-1)

	waveform = np.fromiter(struct.unpack(s, data[35:]), dtype='int8')
	waveform = waveform*dV

	print("filename = {3}, T = {0}, dt = {1}, dV = {2}".format(T, dt, dV, filepath))

	return(dt, dV, waveform)

#%% Чтение файла с энергиями
def read_en(filename_en, line_length=17, col_en=9, col_fon=8, col_trig=6, col_times=1, use_fon = True):
	# Чтение данных из файла с энергиями.
	#Читаем файл с энергиями в массив "слов".
	with open(filename_en,'r') as f:
		file_data_words = f.read().split()

	#Подсчёт количества строк в файле (нужно в случае, если все данные записаны в одну строку).
	lc = int(round(len(file_data_words)/line_length))

	if lc == 0:
		print("File with energies is empty.")
		return(None)
	elif lc <= 2:
		print("Energy file contains less than 3 entries.")
		return(None)

	# Инициализация массивов с времена
	time_en = np.zeros(lc) # Массив времён [c]
	energies = np.zeros(lc, dtype = int) # Массив энергий [попугаи]
	strob = np.zeros(lc, dtype = int) #Массив значений строба

	#Заполняем массивы значениями из файла с энергиями.
	if use_fon == True:
		for i in range(0, lc):
			time_en_list = (file_data_words[i*line_length+col_times]).split("-")
			time_en[i] = float(time_en_list[-1])+ float(time_en_list[-2]) + 60.0*float(time_en_list[-3]) + 3600.0*float(time_en_list[-4])
			energies[i] = int(round(float(file_data_words[i*line_length+col_en]) - float(file_data_words[i*line_length+col_fon])))
			strob[i] = int(round(float(file_data_words[i*line_length+col_trig])))
	else:
		for i in range(0, lc):
			time_en_list = (file_data_words[i*line_length+col_times]).split("-")
			time_en[i] = float(time_en_list[-1])+ float(time_en_list[-2]) + 60.0*float(time_en_list[-3]) + 3600.0*float(time_en_list[-4])
			energies[i] = int(round(float(file_data_words[i*line_length+col_en])))
			strob[i] = int(round(float(file_data_words[i*line_length+col_trig])))
	#Конец чтения файла с энергиями.

	# Поиск момента включения строба.
	# Инициализация переменных
	strob_diff = 0; i_start = 0 # Скачок строба; номер строки, когда включился строб.
	for i in range(1, lc):
		strob_diff_new = strob[i] - strob[i-1] #Текущее значение скачка строба.
		# Если текущее значение больше предыдущего, то текущее присваивается предыдущему.
		if strob_diff_new > strob_diff:
			strob_diff = strob_diff_new
			i_start = i # В i_start записывается текущий номер.
	### TEMP
	'''
	plt.figure(figsize=(10.5, 9.0), dpi=300)
	plt.grid()
	plt.minorticks_on()
	print(time_en[0:i_start+5])
	plt.plot(time_en[0:i_start+5], strob[0:i_start+5], '.', color = 'r')
	plt.plot(time_en[0:i_start+5], strob[0:i_start+5], 'k', color = 'k')
	filename_fig = (os.sep).join(filename_en.split(os.sep)[0:-1])
	print(filename_fig)
	filename_fig_end = filename_en.split(os.sep)[-1]
	filename_fig_end = "strob_"+filename_fig_end.split(".dat")[0]+".png"
	filename_fig = os.path.join(filename_fig, filename_fig_end)
	plt.savefig(filename_fig, bbox_inches='tight')
	plt.close()
	'''
	return(time_en, energies, i_start, lc)

#%% Подсчёт времени по названию файла с люминисценцией/модами.
def calc_lum_time(string):
	'''
	Calculate time from a time string in format h_m_s,ms.
	'''

	if ',' in string:
		string_splitted, ms = string.split(",")
	elif '.' in string:
		string_splitted, ms = string.split(".")
	else:
		print("ERROR: invalid string in calc_lum_time.\nString: \"{}\"".format(string))
		raise TypeError("Invalid string in calc_lum_time.\nString: \"{}\"".format(string))
	digit_num = len(ms)
	if digit_num >=3:
		ms = round(int(ms)/10**(digit_num-3))		
	if '_' in string_splitted:
		h, m, s = [float(f) for f in string_splitted.split("_")]
	else:
		h, m, s = [float(f) for f in string_splitted.split("-")]
	time = h*3600.0 + m*60.0 + s + ms*0.001
	return(time)

def max_find_borders(wf, dt):
	'''
	Defines borders to maximum be searched in (for calibration folders).
	Returns borders in numbers.
	'''

	#Constants.
	ext = '.bin'
	t_start_num = 10 # Отсуп на графике в начале waveform (в связи с наличием провала в начале).
	dt_between_max = 10e-6 #Нижняя граница расстояния между двумя максимумами на wf.
	shift_left = 37.75e-6 #Сдвиг левой границы области усреднения (1-го максимума) относительно главного максимума.
	shift_right = 30.75e-6 #Сдвиг правой границы области усреднения (1-го максимума) относительно главного максимума.

	#%% Пересчёт сдвигов из секунд в отсчёты.
	shift_left_num = int(round(shift_left/dt))
	shift_right_num = int(round(shift_right/dt))
	dt_between_max_num = int(round(dt_between_max/dt))

	#%% Поиск максимумов
	wf_max = np.amax(wf[t_start_num:]) #Величина максимума
	maxima = np.argwhere(wf[t_start_num:] == wf_max).flatten() # Координаты (отсч.) всех точек, значения в которых равны максмиальному.
	if maxima[0] <= shift_left_num: #Проверка того, что максимум не слишком близко к началу выборки.
		energy = 0.0
		return((t_start_num,len(wf)))

	#%% Поиск координаты самого высокого максимума.
	dist = 0.0 #Расстояние между максимумами в отсчётах.
	m0 = maxima[0] #Координата предыдущего максимума (в отсчётах).
	first = maxima[0] # Первый максимум после разрыва.
	first_num = 0 #Номер первого максимума после разрыва.
	last = maxima[0] # Координата последнего максимума в "полочке" (после разрыва, в случае зашкала).
	count = 0 # Подсчёт больших провалов (шире чем dt_between_max).

	for i in range(1, len(maxima)):
		m = maxima[i]
		if m-m0 > dist:
			dist = m-m0
			if dist > dt_between_max_num:
				count += 1
				if count == 2:
					last = m0
					break
			first = m
			first_num = i
		m0 = maxima[i]
	if dist > dt_between_max_num:
		m0_num = first_num
		m_num = first_num+1
		while (m_num < len(maxima)) and (maxima[m_num]-maxima[m0_num] < 2):
			m_num+=1
			m0_num+=1
		if (m_num >= len(maxima)):
			m_num -= 1
			m0_num -= 1
		last = maxima[m_num]
	else:
		first = maxima[0]
		last = maxima[-1]
	max_coord = int(round((last + first)/2)) #Координата самого высокого максимума (в единицах отсчётов).
	#Установка левой границы, правой границы, и границы области для вычисления нулевого значения.
	left_border = max_coord - shift_left_num
	right_border = max_coord - shift_right_num

	return((left_border, right_border)) 

def read_maxima(folder_ac, filenames_ac_times, filenames_ac_info, ext = '.bin', area=(0,1920,0,1200), fon_coeff=1.0, old_osc=False, limit_max=False, inv=False):
	'''
	Функция вычисляет максимум для файлов типа .tif (интерферограммы) и интеграл по площади для файлов типа '.dat' (люминесценция).
	'''

	#%% Константы
	indent = 15
	fon_size = 20 #Размер области для вычисления фона на кадрах с люминесценцией.
	#fon_coeff = 1.0 # Коэффициент, на который домножается фон для определения "содержательных" данных на кадрах с люминесценцией.
	max_level = 4000 # Только данные, не превышающие это значение, будут использованы при сопоставлении (нужно, чтобы отбросить пробой).

	maxima = np.zeros(len(filenames_ac_times))

	if ext == '.bin':
		if limit_max:
			if old_osc:
				for i in range(0,len(filenames_ac_times)):
					dt, dV, wf = read_bin(os.path.join(folder_ac, "__".join(filenames_ac_info[i]) + ext))
					max_pos = max_find_borders(wf, dt)
					if inv:
						maxima[i] = np.amin(wf[max_pos[0]:max_pos[1]])
					else:
						maxima[i] = np.amax(wf[max_pos[0]:max_pos[1]])
			else:
				for i in range(0,len(filenames_ac_times)):
					dt, dV, wf = read_bin_new_program(os.path.join(folder_ac, "__".join(filenames_ac_info[i]) + ext))
					max_pos = max_find_borders(wf, dt)
					if inv:
						maxima[i] = np.amin(wf[max_pos[0]:max_pos[1]])
					else:
						maxima[i] = np.amax(wf[max_pos[0]:max_pos[1]])
		else:
			if old_osc:
				for i in range(0,len(filenames_ac_times)):
					dt, dV, wf = read_bin(os.path.join(folder_ac, "__".join(filenames_ac_info[i]) + ext))
					if inv:
						maxima[i] = np.amin(wf[indent:])
					else:
						maxima[i] = np.amax(wf[indent:])
			else:
				for i in range(0,len(filenames_ac_times)):
					dt, dV, wf = read_bin_new_program(os.path.join(folder_ac, "__".join(filenames_ac_info[i]) + ext))
					if inv:
						maxima[i] = np.amin(wf[indent:])
					else:
						maxima[i] = np.amax(wf[indent:])
	elif ext == '.tif':
		for i in range(0,len(filenames_ac_times)):
			data = plt.imread(os.path.join(folder_ac, "__".join(filenames_ac_info[i]) + ext))
			maxima[i] = np.amax(data)
	elif ext == '.dat' or ext == '.png':
		for i in range(0,len(filenames_ac_times)):
			filename = os.path.join(folder_ac, "__".join(filenames_ac_info[i]) + ext)
			if ext == '.dat':
				try:
					data, width, height = read_dat(filename)
				except struct.error:
					print("Struct error while reading file {}.".format(filename))
					continue
			else:
				data = plt.imread(filename)
			#print(str(filename) + ' ' + str(width) + ' ' + str(height))
			fon = (np.mean(data[0:fon_size, 0:fon_size]) + np.mean(data[0:fon_size, -fon_size:]) + np.mean(data[-fon_size:, 0:fon_size]) + np.mean(data[-fon_size:, -fon_size:]))/4.0
			data = data[area[2]:area[3], area[0]:area[1]]
			data = data-fon_coeff*fon
			#data = data[data > fon_coeff*fon] 
			data = data[data < max_level]
			maxima[i] = np.sum(data)
			#print("fon={}, maxima[i]={}".format(fon, maxima[i]))
	return(maxima)

#%% Формирование списка файлов с акустикой/люминисценцией/интерферометрией.
def make_file_list_to_compare(foldername_ac, ext):
	'''
	Returns array of times in ms, corresponding to files and the files to be compared with energies.
	Arrays are sorted ascending by time.

	Parameters:
		foldername_ac - folder with the files to be compared,
		ext. Allowed values: '.dat', '.bin', '.tif'. In any other case the function returns 1.
	'''

	#Константы
	time_restart_constant = 100e3 # Величина разрыва, при котором считается, что время обнулилось, и акустика начала писаться заново.
	max_time_constant = 3600e3 #1 час - время, после которого происходит обнуление счётчика.

	if (ext != '.dat') and (ext != '.png') and (ext != '.bin') and (ext != '.tif'):
		print('In module "data_proc_basics", function "make_file_list_to_compare":')
		print("ERROR: unknown data type!")
		return(1)
	#Формирование несортированного списка файлов с акустикой.
	if ext == '.tif':
		filenames_ac = [f for f in os.listdir(foldername_ac) if (f.endswith(ext) and "fil" in f)]
	elif ext == '.png':
		filenames_ac = [f for f in os.listdir(foldername_ac) if f.endswith(ext) and '__' in f]
	else:
		filenames_ac = [f for f in os.listdir(foldername_ac) if f.endswith(ext)]
	# Если папка пустая, сразу возвращаем пустые списки (поиск скачка даёт ошибку при пустых списках).
	if filenames_ac == []:
		print("Acoustcs folder is empty.")
		return ([],[])
	filenames_ac_info = [f.split(ext)[0].split("__") for f in filenames_ac]
	filenames_ac_info_ext = []
	if (ext == '.dat') or (ext == '.tif') or (ext == '.png'):
		for f in filenames_ac_info:
			time = calc_lum_time(f[-1])
			filenames_ac_info_ext.append([time, f])
		filenames_ac_info_ext = sorted(filenames_ac_info_ext, key = lambda x: float(x[0])) #Cортированный по первому элементу названия (времени в мс) список файлов с акустикой.
		filenames_ac_times = np.array([int(round(f[0]*1000)) for f in filenames_ac_info_ext])
		filenames_ac_info = [f[1] for f in filenames_ac_info_ext]
	else:
		filenames_ac_info = sorted(filenames_ac_info, key = lambda x: int(x[0])) #Cортированный по первому элементу названия (времени в мс) список файлов с акустикой.
		filenames_ac_times = np.array([int(filename_ac_info[0].split(os.sep)[-1]) for filename_ac_info in filenames_ac_info])

	#Check if there has been time count restart.
	if len(filenames_ac_times) > 1:
		max_diff_num = np.argmax(np.diff(filenames_ac_times))
		if filenames_ac_times[max_diff_num+1] - filenames_ac_times[max_diff_num] > time_restart_constant:
			print("Acoustics tick counter has been reset.")
			filenames_ac_times[0:max_diff_num+1] = filenames_ac_times[0:max_diff_num+1]+max_time_constant
			filenames_ac_times = np.concatenate((filenames_ac_times[max_diff_num+1:], filenames_ac_times[0:max_diff_num+1]))
			filenames_ac_info = filenames_ac_info[max_diff_num+1:]+filenames_ac_info[0:max_diff_num+1]

	return(filenames_ac_times, filenames_ac_info)

#%% Формирование списка файлов с акустикой/люминисценцией/интерферометрией (версия для новой программы записи акустики).
def make_file_list_to_compare_new_program(foldername_ac, ext):
	'''
	Returns array of times in ms, corresponding to files and the files to be compared with energies.
	Arrays are sorted ascending by time.

	Parameters:
		foldername_ac - folder with the files to be compared,
		ext. Allowed values: '.dat', '.bin', '.tif'. In any other case the function returns 1.
	'''

	if (ext != '.dat') and (ext != '.png') and (ext != '.bin') and (ext != '.tif'):
		print('In module "data_proc_basics", function "make_file_list_to_compare":')
		print("ERROR: unknown data type!")
		return(1)
	#Формирование несортированного списка файлов с акустикой.
	if ext == '.tif':
		filenames_ac = [f for f in os.listdir(foldername_ac) if (f.endswith(ext) and "fil" in f)]
	elif ext == '.png':
		filenames_ac = [f for f in os.listdir(foldername_ac) if f.endswith(ext) and '__' in f]
	else:
		filenames_ac = [f for f in os.listdir(foldername_ac) if f.endswith(ext)]
	# Если папка пустая, сразу возвращаем пустые списки (поиск скачка даёт ошибку при пустых списках).
	if filenames_ac == []:
		print("Acoustcs folder is empty.")
		return ([],[])
	filenames_ac_info = [f.split(ext)[0].split("__") for f in filenames_ac]
	filenames_ac_info_ext = []

	for f in filenames_ac_info:
		for segment in f:
			if ('.' in segment or ',' in segment) and '-' in segment:
				break
		time = calc_lum_time(segment)
		filenames_ac_info_ext.append([time, f])
	filenames_ac_info_ext = sorted(filenames_ac_info_ext, key = lambda x: float(x[0])) #Cортированный по первому элементу названия (времени в мс) список файлов с акустикой.
	filenames_ac_times = np.array([int(round(f[0]*1000)) for f in filenames_ac_info_ext])
	filenames_ac_info = [f[1] for f in filenames_ac_info_ext]

	return(filenames_ac_times, filenames_ac_info)

def read_en_all_data(filename_en, line_length=17, col_en=9, col_fon=8, col_trig=6, col_times=1):
	# Чтение данных из файла с энергиями.
	#Читаем файл с энергиями в массив "слов".
	with open(filename_en,'r') as f:
		file_data_words = f.read().split()

	#Подсчёт количества строк в файле (нужно в случае, если все данные записаны в одну строку).
	lc = int(round(len(file_data_words)/line_length))

	if lc == 0:
		print("File with energies is empty.")
		return(None)
	elif lc <= 2:
		print("Energy file contains less than 3 entries.")
		return(None)

	# Инициализация массивов с времена
	time_en = np.zeros(lc) # Массив времён [c]
	energies = np.zeros(lc, dtype = int) # Массив энергий [попугаи]
	signal = np.zeros(lc, dtype = int)
	fon = np.zeros(lc, dtype = int)
	strob = np.zeros(lc, dtype = int) #Массив значений строба

	#Заполняем массивы значениями из файла с энергиями.
	for i in range(0, lc):
		time_en_list = (file_data_words[i*line_length+col_times]).split("-")
		time_en[i] = float(time_en_list[-1])+ float(time_en_list[-2]) + 60.0*float(time_en_list[-3]) + 3600.0*float(time_en_list[-4])
		energies[i] = int(round(float(file_data_words[i*line_length+col_en]) - float(file_data_words[i*line_length+col_fon])))
		signal[i] = int(round(float(file_data_words[i*line_length+col_en])))
		fon[i] = int(round(float(file_data_words[i*line_length+col_fon])))
		strob[i] = int(round(float(file_data_words[i*line_length+col_trig])))
	#Конец чтения файла с энергиями.

	# Поиск момента включения строба.
	# Инициализация переменных
	strob_diff = 0; i_start = 0 # Скачок строба; номер строки, когда включился строб.
	for i in range(1, lc):
		strob_diff_new = strob[i] - strob[i-1] #Текущее значение скачка строба.
		# Если текущее значение больше предыдущего, то текущее присваивается предыдущему.
		if strob_diff_new > strob_diff:
			strob_diff = strob_diff_new
			i_start = i # В i_start записывается текущий номер.
	return(time_en, energies, i_start, lc, signal, fon)
