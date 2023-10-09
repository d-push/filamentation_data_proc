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

	return(T, dt, dV, waveform)

#%% Чтение бинарных файлов с нового осциллографа.	
def read_bin_new_Rudnev(filepath):
	try:
		with open(filepath, "rb") as binary_file:
			# Read the whole file at once
			data = binary_file.read()
	except (struct.error):
		return(None)

	E = struct.unpack('>l', data[0:4])[0]
	T = (struct.unpack('15s', data[4:19])[0]).decode('UTF-8')
	dt = struct.unpack('>d', data[19:27])[0]
	ch0_on = struct.unpack('?', data[27:28])[0] #whether channel 0 was on
	dV0 = struct.unpack('>d', data[28:36])[0] #[mV/bit]
	ch0_adj = struct.unpack('>d', data[36:44])[0] #channel0 adjustment
	ch1_on = struct.unpack('?', data[44:45])[0] #whether channel 1 was on
	dV1 = struct.unpack('>d', data[45:53])[0] #[mV/bit]
	ch1_adj = struct.unpack('>d', data[53:61])[0] #channel0 adjustment
	
	wf_len = len(data)-61
	s = '>b'+ 'b'*(wf_len-1)
	
	try:
		waveform = np.fromiter(struct.unpack(s, data[61:]), dtype='int8')
	except (struct.error):
		return(None)
	
	print(f"filename = {filepath}, E = {E}, T = {T}, dt = {dt}, dV0 = {dV0}, dV1 = {dV1}")
	
	two_channels = False
	if ch0_on and ch1_on:
		ch_num = 2
		wf0 = waveform[0::2]*dV0-ch0_adj
		wf1 = waveform[1::2]*dV1-ch1_adj
		two_channels = True
		return(two_channels, T, dt, wf0, wf1)
	elif ch0_on:
		ch_num = 0
		waveform = waveform*dV0-ch0_adj
		two_channels = False
		return(two_channels, T, dt, ch0_arr)
	elif ch1_on:
		ch_num = 1
		waveform = waveform*dV1-ch1_adj
		two_channels = False
		return(two_channels, T, dt, ch1_arr)
	else:
		print("ERROR. Unknown channels configuration in read_bin_new_Rudnev in data_proc_basics_script")
		return("ERROR")

def read_bin_basing_on_args(filename_ac, args=None):
	'''Read binary file depending on what program was used to write files.'''
	
	if args.old_osc:
		dt, dV, wf = read_bin(filename_ac)
		two_channels = False
	elif args.new_Rud:
		wf_data = read_bin_new_Rudnev(filename_ac)
		if wf_data is None:
			return(None)
		two_channels = wf_data[0]
		if two_channels:
			T, dt, wf0, wf1 = wf_data[1:]
			wf0 = wf0[2:]
			wf1 = wf1[2:]
		else:
			T, dt, wf = wf_data[1:]
			wf = wf[2:]
	else:
		two_channels = False
		dt, dV, wf = read_bin_new_program(filename_ac)
	
	if two_channels:
		wfs = [wf0, wf1]
		return(two_channels, dt, wfs)
	else:
		wfs = [wf]
		return(two_channels, dt, wfs)
	
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
	elif string.isdecimal():
		return(int(string))
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
	
def calc_RAW_time(filename_ac_info, seconds_to_subtract):
	'''
	Calculate time from a time string in format like "Camera MV-UB130GM#68E65449-2-Snapshot-20210925220546-391516957331.RAW"
	'''
	
	hours = (int(filename_ac_info[-2]) % 10**6)
	seconds = hours % 10**2
	hours = hours // 10**2
	minutes = hours % 10**2
	hours = hours // 10**2
	time = (float(filename_ac_info[-1])/10**7 - seconds_to_subtract) + hours*3600 + minutes*60 + seconds
	
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

def read_maxima(folder_ac, filenames_ac_times, filenames_ac_info, ext, bd_mult, bd_single, area=(0,1920,0,1200), fon_coeff=1.0, old_osc=False, limit_max=False, inv=False, ac_lims=None, use_run_av=False, av_params=(11,20), bg_from_empty=None, subtr_plane=False):

	'''
	Функция вычисляет максимум для файлов типа .tif (интерферограммы) и интеграл по площади для файлов типа '.dat' (люминесценция).

	Параметры:
	limit_max: True - ограничить диапазон поиска максимума границами, найденными функцией max_find_borders (для файлов калибровки).
	           False - искать в диапазоне при i>indent.
	ac_lims: массив из двух элементов (ndarray), содержащий начало и конец области, в которой будет осуществляться поиск максимума (для акустики), либо None.
	use_run_av: использовать или не использовать бегущее среднее.
	'''

	#%% Константы
	indent = 15 #default 15
	fon_size = 20 #Размер области для вычисления фона на кадрах с люминесценцией.
	#fon_coeff = 1.0 # Коэффициент, на который домножается фон для определения "содержательных" данных на кадрах с люминесценцией.
	max_level = 4000 # Только данные, не превышающие это значение, будут использованы при сопоставлении (нужно, чтобы отбросить пробой).

	maxima = np.zeros(len(filenames_ac_times))

	if ext == '.bin':
		for i in range(0,len(filenames_ac_times)):
			if old_osc:
				dt, dV, wf = read_bin(os.path.join(folder_ac, "__".join(filenames_ac_info[i]) + ext))
			else:
				dt, dV, wf = read_bin_new_program(os.path.join(folder_ac, "__".join(filenames_ac_info[i]) + ext))
			if limit_max:
				max_pos = max_find_borders(wf, dt)
			elif ac_lims is None:
				max_pos = [indent, len(wf)]
			else:
				if ac_lims[1]/dt > len(wf):
					print("\nWARNING: ac_lims exceeds waveform length!\n")
				max_pos = [int(round(ac_lims[0]/dt)), min(int(round(ac_lims[1]/dt)), len(wf))]
			wf = wf[max_pos[0]:max_pos[1]]
			if use_run_av:
				for av_par in av_params:
					wf = run_av(wf, window = av_par)
			if inv:
				maxima[i] = np.amin(wf)
			else:
				maxima[i] = np.amax(wf)
	#elif ext == '.tif':
	#	for i in range(0,len(filenames_ac_times)):
	#		data = plt.imread(os.path.join(folder_ac, "__".join(filenames_ac_info[i]) + ext))
	#		maxima[i] = np.amax(data)
	elif ext == '.RAW':
		for i in range(0,len(filenames_ac_times)):
			filename = os.path.join(folder_ac, "-".join(filenames_ac_info[i]) + ext)
			data_info = read_modes_basing_on_ext(filename, ext)
			if not data_info:
				continue
			else:
				data, width, height = data_info
			if bd_mult and bd_single:
				data = apply_bd_map(data, bd_mult, bd_single)
			if subtr_plane:
				data = subtract_plane(data)
			if bg_from_empty is not None:
				data = data - bg_from_empty
			maxima[i] = np.sum(data)
	elif ext == '.dat' or ext == '.txt' or ext == '.png' or ext == '.tif':
		for i in range(0,len(filenames_ac_times)):			
			filename = os.path.join(folder_ac, "__".join(filenames_ac_info[i]) + ext)
			data_info = read_modes_basing_on_ext(filename, ext)
			if not data_info:
				maxima[i] = None
				continue
			else:
				data, width, height = data_info
			if len(bd_mult) and len(bd_single):
				data = apply_bd_map(data, bd_mult, bd_single)
			if subtr_plane:
				data = subtract_plane(data)
			if bg_from_empty is not None:
				data = data - bg_from_empty
			maxima[i] = np.sum(data)
	else:
		print("ERROR in function read_maxima: unknown file type")
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

	if (ext != '.dat') and (ext != '.png') and (ext != '.bin') and (ext != '.tif') and (ext != '.RAW'):
		print('In module "data_proc_basics", function "make_file_list_to_compare":')
		print("ERROR: unknown data type!")
		return(1)
	#Формирование несортированного списка файлов с акустикой.
	if ext == '.tif':
		filenames_ac = [f for f in os.listdir(foldername_ac) if (f.endswith(ext) and "fil" in f)]
	elif ext == '.png':
		filenames_ac = [f for f in os.listdir(foldername_ac) if f.endswith(ext) and '__' in f]
	elif ext == '.RAW':
		filenames_ac = [f for f in os.listdir(foldername_ac) if f.endswith(ext) and 'MV-UB130GM' in f]
	else:
		filenames_ac = [f for f in os.listdir(foldername_ac) if f.endswith(ext)]
	# Если папка пустая, сразу возвращаем пустые списки (поиск скачка даёт ошибку при пустых списках).
	if filenames_ac == []:
		print("Acoustcs folder is empty.")
		return ([],[])
	if ext == '.RAW':
		filenames_ac_info = [f.split(ext)[0].split("-") for f in filenames_ac]
	else:
		filenames_ac_info = [f.split(ext)[0].split("__") for f in filenames_ac]
	filenames_ac_info_ext = []
	if (ext == '.dat') or (ext == '.tif') or (ext == '.png'):
		for f in filenames_ac_info:
			time = calc_lum_time(f[-1])
			filenames_ac_info_ext.append([time, f])
		filenames_ac_info_ext = sorted(filenames_ac_info_ext, key = lambda x: float(x[0])) #Cортированный по первому элементу названия (времени в мс) список файлов с акустикой.
		filenames_ac_times = np.array([int(round(f[0]*1000)) for f in filenames_ac_info_ext])
		filenames_ac_info = [f[1] for f in filenames_ac_info_ext]
	elif (ext == '.RAW'):
		times_from_tail = np.array([int(f[-1]) for f in filenames_ac_info])
		seconds_to_subtract = int(np.floor(np.amin(times_from_tail)/10**7))*10**3
		for f in filenames_ac_info:
			hours = (int(f[-2]) % 10**6)
			seconds = hours % 10**2
			hours = hours // 10**2
			minutes = hours % 10**2
			hours = hours // 10**2
			time = int(round( (float(f[-1])/10**4 - seconds_to_subtract) )) + hours*3600 + minutes*60 + seconds
			filenames_ac_info_ext.append([time, f])
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

	if (ext != '.dat') and (ext != '.png') and (ext != '.RAW') and (ext != '.bin') and (ext != '.tif'):
		print('In module "data_proc_basics", function "make_file_list_to_compare_new_program":')
		print("ERROR: unknown data type!")
		return(1)
	#Формирование несортированного списка файлов с акустикой.
	if ext == '.tif':
		filenames_ac = [f for f in os.listdir(foldername_ac) if (f.endswith(ext) and "fil" in f)]
	elif ext == '.png':
		filenames_ac = [f for f in os.listdir(foldername_ac) if f.endswith(ext) and '__' in f]
	elif ext == '.RAW':
		filenames_ac = [f for f in os.listdir(foldername_ac) if f.endswith(ext) and 'MV-UB130GM' in f]
	else:
		filenames_ac = [f for f in os.listdir(foldername_ac) if f.endswith(ext)]
		
	if filenames_ac == [] and ext =='.png':
		filenames_ac = [f for f in os.listdir(foldername_ac) if f.endswith(ext) and ('_' in f) and ('.' in f.split(ext)[0])]
		
	# Если папка пустая, сразу возвращаем пустые списки (поиск скачка даёт ошибку при пустых списках).
	if filenames_ac == []:
		print("Acoustcs folder is empty.")
		return ([],[])
	if ext == '.RAW':
		filenames_ac_info = [f.split(ext)[0].split("-") for f in filenames_ac]
	else:
		filenames_ac_info = [f.split(ext)[0].split("__") for f in filenames_ac]
	filenames_ac_info_ext = []

	if ext == '.RAW':
		filenames_ac_info = sorted(filenames_ac_info, key = lambda x: int(x[-1]))
		
		time_mode = np.zeros(len(filenames_ac_info))
		nonzero_shift = True
		shift = 0
		step = 1
		while nonzero_shift:
			nonzero_shift = False
			for i in range(0, len(filenames_ac_info)):
				time_mode[i] = shift + (
                        float(filenames_ac_info[i][-1]) - float(filenames_ac_info[0][-1])) / 1e7 + float(
                    filenames_ac_info[0][-2][-2:]) + 60 * float(filenames_ac_info[0][-2][-4:-2]) + 3600 * float(
                    filenames_ac_info[0][-2][-6:-4])
				if (int(time_mode[i]) < int(filenames_ac_info[i][-2][-2:]) + 60 * int(
                        filenames_ac_info[i][-2][-4:-2]) + 3600 * int(filenames_ac_info[i][-2][-6:-4])):
					shift = shift + 1 / (2 ** step)
					step = step + 1
					nonzero_shift = True
				if (int(time_mode[i]) > int(filenames_ac_info[i][-2][-2:]) + 60 * int(
						filenames_ac_info[i][-2][-4:-2]) + 3600 * int(filenames_ac_info[i][-2][-6:-4])):
					shift = shift - 1 / (2 ** step)
					step = step + 1
					nonzero_shift = True
					
		filenames_ac_info_ext = np.array(list(zip(time_mode, filenames_ac_info)))
	else:
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

def autodetect_en_line_length(filename_en):
	#%%Константы
	max_length = 40 #Maximal possible line length.

	#Читаем первые 2 строки из файла с энергиями.
	with open(filename_en,'r') as f:
		first_string = f.readline()
		next_string = f.readline()

	length = len(first_string.split())
	if length <= max_length and next_string != '':
		return length
	else:
		return "FAIL"

def read_en_all_data(filename_en, line_length=17, col_en=9, col_fon=8, col_trig=6, col_times=1):
	# Чтение данных из файла с энергиями.
	#Читаем файл с энергиями в массив "слов".
	with open(filename_en,'r') as f:
		file_data_words = f.read().split()

	#Подсчёт количества строк в файле (нужно в случае, если все данные записаны в одну строку).
	lc = int(round(len(file_data_words)/line_length))

	if lc == 0:
		print("\n WARNING: File with energies is empty.\n")
		raise EmptyEnFile
		return(None)
	elif lc <= 2:
		print("\n WARNING: Energy file contains less than 3 entries.\n")
		raise EmptyEnFile
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
