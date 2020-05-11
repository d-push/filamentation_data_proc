#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:57:52 2019

@author: dmitrii
"""

import os
import numpy as np
import shutil as sh
import data_proc.data_proc_basics.data_proc_basics_script as dpb
#import matplotlib.pyplot as plt

#import calibrator_1 as calibr

#%% Constants
empty_threshold = 350.0
n_ac_try_range = 10
is_zero_int = 0.1 #Parameter for numbers comparation which should have integer values.

def init_shifts(shift, ext, i_start):
	'''
	Функция написана исходя из того, что в программе сопоставления текущая энергия сдвинута на 1 относительно акустики.
	'''
	if shift >= 1:
		j = shift # Инициализация счётчика для акустики.
	else:
		j = 0
	if ext == '.bin'or ext == '.dat' or ext == '.tif':
		if shift >= 0:
			i = i_start-1 #Инициализация счётчика для энергии.
		else:
			i = i_start-1-shift
	else:
		print("ERROR: unknown extension!")
		return("ERROR")
	return((i,j))
	#print("i = {}, i_start = {}, j = {}".format(i, i_start, j))

def shift_search(filenames_ac_times, dt = 0.1, shift_border_min = -3, shift_border_max = 5):
	#%% Поиск "правильного" сдвига.
	i = 0
	time_n = filenames_ac_times[i] + dt*1000
	time_nn = filenames_ac_times[i+1] + dt*1000
	while (filenames_ac_times[i+1] >= time_n + shift_border_max) or (filenames_ac_times[i+1] <= time_n + shift_border_min) or (filenames_ac_times[i+2] >= time_nn + shift_border_max) or (filenames_ac_times[i+2] <= time_nn + shift_border_min):
		i+=1
		time_n = filenames_ac_times[i] + dt*1000
		time_nn = filenames_ac_times[i+1] + dt*1000

	shift = i
	return(shift)

#%% Проверка, были ли пропущены стробы.
def check_if_there_were_lost(filename_en, foldername_ac, ext='.bin', col_en=9, col_fon=8, col_trig=6, col_times=1, line_length=17, use_trig = True, use_fon = True, old_osc=False):

	#%% Constants
	threshold = 80 #[ms] - допустимая разница между ожидаемым и реальным длиной выборки по времени. Если больше - считаем, что пропущен кадр.
	DELTA_en = 0.15 #[s] - максимальная допустимая разница между временами фиксации соседних значений энергий в выборке.

	#%% Read parameters from the energy file.
	en_param_list = dpb.read_en(filename_en, line_length, col_en, col_fon, col_trig, col_times, use_fon = use_fon)
	if en_param_list == None: #Если массив энергий пуст, завершаем выполнение функции.
		return("Empty.")
	else:
		time_en, energies, i_start, lc = en_param_list
	if use_trig == False:
		i_start = 1

	#%% Read list of filenames with luminescence.
	if old_osc:
		filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	else:
		try:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare_new_program(foldername_ac, ext)
		except ValueError:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	if len(filenames_ac_times) == 0: #Если папка с акустикой пуста, сразу завершаем выполнение функции.
		return("Empty")

	#%% Проверка, если ли "подвисания" при записи энергий и пропуски кадров.
	en_diff = np.diff(time_en)

	if abs((filenames_ac_times[-1] - filenames_ac_times[0]) - (len(filenames_ac_times)-1)*100.0) < threshold and np.all(en_diff[en_diff < DELTA_en]):
		print("Время по началу и концу выборки: {}, время по длине выборки: {}".format(filenames_ac_times[-1] - filenames_ac_times[0], len(filenames_ac_times)*100.0))
		return(True)
	else:
		if abs((filenames_ac_times[-1] - filenames_ac_times[0]) - len(filenames_ac_times)*100.0) >= threshold:
			print("Время по началу и концу выборки: {}, время по длине выборки: {}".format(filenames_ac_times[-1] - filenames_ac_times[0], len(filenames_ac_times)*100.0))
			print("Были пропущены файлы с акустикой")
		if np.all(en_diff >= DELTA_en):
			print("Были сбои в записи энергии")
		return(False)

#%% Функция сопоставления акустики и энергии.
def compare(filename_en, foldername_ac, folder_to_write, ext='.bin', col_en=9, col_fon=8, col_trig=6, col_times=1, line_length=17, dt=0.1, EPS_TIME_EN=0.12, EPS_TIME_AC=5, N_try=5, EPS_TIME_EN_1=0.025, use_trig = True, shift = None, use_fon = True, old_osc=False):

	'''
	This function matches energies from a file with energies with other data obtained in single-shot regime (acoustics, or modes, or luminescence, or interferograms) in "parrots".

	Parameters:
	filename_en - energy file,
	foldername_ac - folder with files to be compared with the energies from the energy file,
	folder_to_write - path for the files to be written,
	ext='.bin' - extension for data filenames ('.bin' is default),
	col_en=9 - number of column that contain energy values from photodiode (in the energy files),
	col_fon=8 - number of column that contain background values from photodiode (in the energy files),
	col_trig=6 - number of column that contain strobe values (the data when the trigger is turned on),
	col_times=1 - column with times, when the energies have been written into the file,
	line_length=17 - number of columns in energy file,
	dt=0.1 - laser (and sinchropulse) repetition rate,
	shift=-2 - strange shift between the energies and data. In the program the shift is added by +1.

	Program constants:
	EPS_TIME_EN=0.14
	EPS_TIME_AC=120
	N_try=5 - number of pulses after which the function checks if the steps in energy file restored after a strob has been missed,
	EPS_TIME_EN_1=0.025 - if the difference between the calculated and real times within N_try shots is less than this value, shots are considered to be haven't been missed

	WARNING: program parameters are fitted for the nice program work. It is NOT recommended to change them.
	'''

	#%% Инициализация переменных
	count_en_skipped = 0
	count_en_corrected = 0
	dt_ms = dt*1000

	#%% Read parameters from the energy file.
	en_param_list = dpb.read_en(filename_en, line_length, col_en, col_fon, col_trig, col_times, use_fon = use_fon)
	if en_param_list == None: #Если массив энергий пуст, завершаем выполнение функции.
		return("Empty.")
	else:
		time_en, energies, i_start, lc = en_param_list
	if use_trig == False:
		print("Trigger column is not used.")
		i_start = 1
	#%% Read list of filenames with luminescence.
	if old_osc:
		filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	else:
		try:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare_new_program(foldername_ac, ext)
		except ValueError:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	if len(filenames_ac_times) == 0: #Если папка с акустикой пуста, сразу завершаем выполнение функции.
		return("Empty")

	#%% Поиск "правильного" сдвига.
	if shift == None:
		shift = shift_search(filenames_ac_times, dt = 0.1, shift_border_min = -3, shift_border_max = 5)
	#shift = 0


	#%% Цикл, сопоставляющий акустику и энергию.
	i,j = init_shifts(shift, ext, i_start)

	while (i < lc - 1) and (j < len(filenames_ac_info)):
		if i > i_start and j > shift and j>0:
			delta_time_en = abs(time_en[i+1] - time_en[i] - dt)
			delta_time_ac = abs(filenames_ac_times[j] - filenames_ac_times[j-1] - dt_ms)
		else:
			delta_time_en = 0
			delta_time_ac = 0

		#Проверка на пропущенные стробы в энергиях и акустике.
		if delta_time_en < EPS_TIME_EN and delta_time_ac < EPS_TIME_AC:
			if ext=='.RAW':
				filename_ac_new = os.path.join(folder_to_write, str(energies[i+1]) + "_" + str(i+1) + "_" + "-".join(filenames_ac_info[j]) + ext)
				filename_ac = os.path.join(foldername_ac, "-".join(filenames_ac_info[j])+ext)
			else:
				filename_ac_new = os.path.join(folder_to_write, str(energies[i+1]) + "_" + str(i+1) + "_" + "__".join(filenames_ac_info[j]) + ext)
				filename_ac = os.path.join(foldername_ac, "__".join(filenames_ac_info[j])+ext)
			sh.copyfile(filename_ac, filename_ac_new)
			i += 1; j += 1
		elif delta_time_ac >= EPS_TIME_AC and delta_time_en < EPS_TIME_EN:
			#print("WARNING: acoustics was skipped.")
			#delta_t_ac = int(round(delta_time_ac/100.0))
			print("WARNING: acoustics was skipped")
			#time_ac_j_minus_2 = np.mean(time_en[i-:i]) + dt
			N_ac_try = 2; success = False
			if j < 1:
				j_new = 1
			else:
				j_new = j
			while (j + N_ac_try < len(filenames_ac_info)):
				n_ac_try = 1
				while (n_ac_try < N_ac_try + n_ac_try_range) and (j_new + n_ac_try < len(filenames_ac_info)):
					if i+(j_new-j)+n_ac_try >= len(time_en):
						success = False
						print("File with energies finished before acoustic files ended.")
					break
					if abs(filenames_ac_times[j_new+N_ac_try-1] - filenames_ac_times[j_new-1] - dt*n_ac_try*1000) < EPS_TIME_AC:
						if abs(time_en[i+(j_new-j)+n_ac_try] - time_en[i+(j_new-j)] - dt*n_ac_try) < EPS_TIME_EN_1:
							i = i-1+(j_new-j)+n_ac_try; j = j_new + N_ac_try-1
							if ext=='.RAW':
								filename_ac_new = os.path.join(folder_to_write, str(energies[i+1]) + "_" + str(i+1) + "_" + "-".join(filenames_ac_info[j]) + ext)
							else:
								filename_ac_new = os.path.join(folder_to_write, str(energies[i+1]) + "_" + str(i+1) + "_" + "__".join(filenames_ac_info[j]) + ext)
							sh.copyfile(filename_ac, filename_ac_new)
							print("Acoustics has been successfully corrected. {} acoustic files has been skipped".format(N_ac_try-1))
							success = True
							i += 1; j += 1
							break
						else:
							print("WARNING: energy has been skipped.")
							n_en_try = 1 #n_en_try должно быть больше 0, иначе цикл будет завершаться после 1-й итерации.
							while (n_en_try < 2*n_ac_try) and (i + (j_new-j) + n_en_try < len(time_en)):
								if abs(time_en[i+(j_new-j)+n_en_try] - time_en[i+(j_new-j)] - dt*n_ac_try) < EPS_TIME_EN_1:
									i = i-1+(j_new-j)+n_ac_try; j = j_new + N_ac_try-1
									if ext=='.RAW':
										filename_ac_new = os.path.join(folder_to_write, str(energies[i+1]) + "_" + str(i+1) + "_" + "-".join(filenames_ac_info[j]) + ext)
									else:
										filename_ac_new = os.path.join(folder_to_write, str(energies[i+1]) + "_" + str(i+1) + "_" + "__".join(filenames_ac_info[j]) + ext)
									sh.copyfile(filename_ac, filename_ac_new)
									print("Acoustics has been successfully corrected. {} acoustic files has been skipped".format(N_ac_try-1))
									print("Energy has been successfully corrected.")
									success = True
									i += 1; j += 1
									break
								n_en_try += 1
							if not success: #Если произошёл выход из цикла, значит хотя бы одно из условий выхода было выполнено. При этом успех не был достигнут.
								print("WARNING: acoustics and energy was NOT corrected.")
							break
					else:
						#print("N_ac_try = {}, n_ac_try = {}, filenames_ac_times[j_new+N_ac_try-1] = {}, filenames_ac_times[j_new-1] = {}".format(N_ac_try, n_ac_try, filenames_ac_times[j_new+N_ac_try-1], filenames_ac_times[j_new-1]))
						n_ac_try += 1
				if success:
					break
				else:
					N_ac_try += 1
			if not success:
				print("WARNING: acoustics has NOT been successfully corrected!")
				print(foldername_ac)
				if ext=='.RAW':
					print("filename_ac_j = {}".format("-".join(filenames_ac_info[j])))
				else:
					print("filename_ac_j = {}".format("__".join(filenames_ac_info[j])))
				break
		else:
			print("WARNING: suspicious energy time behavior.")
			#delta_j = 1
			time_en_i_minus_1 = np.mean(time_en[i-3:i]) + dt
			if (i+N_try < lc) and (j < len(filenames_ac_info)):
				if abs(time_en[i+N_try-1] - time_en_i_minus_1 - dt*N_try) < EPS_TIME_EN_1:
					print("Energy was not skipped.")
					# i_1, j_1 - cчётчики кадров для энергии и акустики для внутреннего цикла
					for i_1 in range(i,i+N_try-1):
						if j<len(filenames_ac_info):
							if ext=='.RAW':
								filename_ac_new = os.path.join(folder_to_write, str(energies[i+1]) + "_" + str(i+1) + "_" + "-".join(filenames_ac_info[j]) + ext)
								filename_ac = os.path.join(foldername_ac, "-".join(filenames_ac_info[j]) + ext)
							else:
								filename_ac_new = os.path.join(folder_to_write, str(energies[i+1]) + "_" + str(i+1) + "_" + "__".join(filenames_ac_info[j]) + ext)
								filename_ac = os.path.join(foldername_ac, "__".join(filenames_ac_info[j]) + ext)
							sh.copyfile(filename_ac, filename_ac_new)
							i += 1
							j += 1
						else:
							break
				else:
					count_en_skipped += 1
					n_try = N_try+1
					while (n_try <= 2*N_try) and abs(time_en[i+N_try-1] - time_en_i_minus_1 - dt*n_try) >= EPS_TIME_EN_1:
						n_try += 1
					if abs(time_en[i+N_try-1] - time_en_i_minus_1 - dt*n_try) < EPS_TIME_EN_1:
						print("Energy was skipped {} times".format(n_try-N_try))
						print("Acoustics will be skipped {} times".format(n_try))
						i += N_try - 1
						j += n_try - 1
						count_en_corrected += 1
						print("Energy has been successfully corrected.")
					else:
						print("WARNING: energy was NOT successfully corrected!")
						break

	print("Energy was skipped {0} times, {1} times successfully corrected.".format(count_en_skipped, count_en_corrected))
	if count_en_skipped - count_en_corrected > 0:
		print("WARNING: energy was not corrected {0} times.".format(count_en_skipped - count_en_corrected))
	print("shift = {}".format(shift))
	return("OK")

#############

def compare_new_method(filename_en, foldername_ac, folder_to_write, ext='.bin', shift = None, col_en=9, col_fon=8, col_trig=6, col_times=1, line_length=17, dt=0.1, DELTA = 15, use_trig = True, use_fon = True, old_osc=False):

	#%% Read parameters from the energy file.
	en_param_list = dpb.read_en(filename_en, line_length, col_en, col_fon, col_trig, col_times, use_fon = use_fon)
	if en_param_list == None: #Если массив энергий пуст, завершаем выполнение функции.
		return("Empty.")
	else:
		time_en, energies, i_start, lc = en_param_list
	if use_trig == False:
		print("Trigger column is not used.")
		i_start = 1

	#%% Read list of filenames with luminescence.
	if old_osc:
		filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	else:
		try:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare_new_program(foldername_ac, ext)
		except ValueError:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	if len(filenames_ac_times) == 0: #Если папка с акустикой пуста, сразу завершаем выполнение функции.
		return("Empty")

	#%% Поиск "правильного" сдвига.
	if shift == None:
		shift = shift_search(filenames_ac_times, dt = 0.1, shift_border_min = -3, shift_border_max = 5)
	#shift = 0

	#%% Цикл, сопоставляющий акустику и энергию.
	if shift >= 0:
		time_ac_zero = filenames_ac_times[shift]
		time_en_zero = time_en[i_start]
		shift_i = i_start
		shift_j = shift
	else:
		time_ac_zero = filenames_ac_times[0]
		time_en_zero = time_en[i_start-shift]
		shift_i = i_start-shift
		shift_j = 0

	time_en_new = (np.around((time_en - time_en_zero)*1000)).astype(int) #Времена для энергий в мс, сдвинутые в начало отсчёта.
	time_ac_new = filenames_ac_times - time_ac_zero

	time_en_new = time_en_new[time_en_new > -is_zero_int]
	time_ac_new = time_ac_new[time_ac_new > -is_zero_int]

	start_ac_count = 0
	for i in range(0,len(time_en_new)):
		for j in range(start_ac_count, len(time_ac_new)):
			#if i==1 and j in range(0,3):
				#print("!!! i={}, j={}, time_en_new[i] = {}, time_ac_new[j]={}, abs={}".format(i, j, time_en_new[i], time_ac_new[j],abs(time_en_new[i]-time_ac_new[j])))
			if abs(time_en_new[i]-time_ac_new[j]) < DELTA:
				if ext=='.RAW':
					filename_ac_new = os.path.join(folder_to_write, str(energies[i+shift_i]) + "_" + str(i+shift_i) + "_" + "-".join(filenames_ac_info[j+shift_j]) + ext)
					filename_ac = os.path.join(foldername_ac, "-".join(filenames_ac_info[j+shift_j]) + ext)
				else:
					filename_ac_new = os.path.join(folder_to_write, str(energies[i+shift_i]) + "_" + str(i+shift_i) + "_" + "__".join(filenames_ac_info[j+shift_j]) + ext)
					filename_ac = os.path.join(foldername_ac, "__".join(filenames_ac_info[j+shift_j]) + ext)
				sh.copyfile(filename_ac, filename_ac_new)
				start_ac_count = j
				break
	return("OK")

#%% Функция сопоставления акустики и энергии.
def compare_not_save(filename_en, foldername_ac, ext='.bin', shift = 0, col_en=9, col_fon=8, col_trig=6, col_times=1, line_length=17, dt=0.1, EPS_TIME_EN=0.12, EPS_TIME_AC=12, N_try=5, EPS_TIME_EN_1=0.025, use_trig = True, use_fon = True, old_osc=False):

	#%% Инициализация переменных
	count_en_skipped = 0
	count_en_corrected = 0
	dt_ms = dt*1000

	#%% Read parameters from the energy file.
	en_param_list = dpb.read_en(filename_en, line_length, col_en, col_fon, col_trig, col_times, use_fon = use_fon)
	if en_param_list == None: #Если массив энергий пуст, завершаем выполнение функции.
		return("Empty.")
	else:
		time_en, energies, i_start, lc = en_param_list
	if use_trig == False:
		print("Trigger column is not used.")
		i_start = 1

	calibration = np.zeros((len(energies), 2))
	#%% Read list of filenames with luminescence.
	if old_osc:
		filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	else:
		try:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare_new_program(foldername_ac, ext)
		except ValueError:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	if len(filenames_ac_times) == 0: #Если папка с акустикой пуста, сразу завершаем выполнение функции.
		return("Empty")

	#%% Поиск "правильного" сдвига.
	#shift = shift_search(filenames_ac_times, dt = 0.1, shift_border_min = -3, shift_border_max = 5)
	#shift = 0

	#%% Цикл, сопоставляющий акустику и энергию.

	i,j = init_shifts(shift, ext, i_start)
	while (i < lc - 1) and (j < len(filenames_ac_info)):
		if i > i_start and j > shift and j>0:
			delta_time_en = abs(time_en[i+1] - time_en[i] - dt)
			delta_time_ac = abs(filenames_ac_times[j] - filenames_ac_times[j-1] - dt_ms)
		else:
			delta_time_en = 0
			delta_time_ac = 0

		#Проверка на пропущенные стробы в энергиях и акустике.
		if delta_time_en < EPS_TIME_EN and delta_time_ac < EPS_TIME_AC:
			calibration[i,0] = energies[i+1]
			calibration[i,1] = filenames_ac_times[j]
			i += 1; j += 1
		elif delta_time_ac >= EPS_TIME_AC and delta_time_en < EPS_TIME_EN:
			#print("WARNING: acoustics was skipped")
			N_ac_try = 2; success = False
			if j < 1:
				j_new = 1
			else:
				j_new = j
			while (j + N_ac_try < len(filenames_ac_info)):
				n_ac_try = 1
				while (n_ac_try < N_ac_try + n_ac_try_range) and (j_new + n_ac_try < len(filenames_ac_info)):
					if i+(j_new-j)+n_ac_try >= len(time_en):
						success = False
						print("File with energies finished before acoustic files ended.")
					break
					if abs(filenames_ac_times[j_new+N_ac_try-1] - filenames_ac_times[j_new-1] - dt*n_ac_try*1000) < EPS_TIME_AC:
						if abs(time_en[i+(j_new-j)+n_ac_try] - time_en[i+(j_new-j)] - dt*n_ac_try) < EPS_TIME_EN_1:
							i = i-1+(j_new-j)+n_ac_try; j = j_new + N_ac_try-1
							calibration[i,0] = energies[i+1]
							calibration[i,1] = filenames_ac_times[j]
							print("Acoustics has been successfully corrected. {} acoustic files has been skipped".format(N_ac_try-1))
							success = True
							i += 1; j += 1
							break
						else:
							print("WARNING: energy has been skipped.")
							n_en_try = 1 #n_en_try должно быть больше 0, иначе цикл будет завершаться после 1-й итерации.
							while (n_en_try < 2*n_ac_try) and (i + (j_new-j) + n_en_try < len(time_en)):
								if abs(time_en[i+(j_new-j)+n_en_try] - time_en[i+(j_new-j)] - dt*n_ac_try) < EPS_TIME_EN_1:
									i = i-1+(j_new-j)+n_ac_try; j = j_new + N_ac_try-1
									calibration[i,0] = energies[i+1]
									calibration[i,1] = filenames_ac_times[j]
									#print("Acoustics has been successfully corrected. {} acoustic files has been skipped".format(N_ac_try-1))
									print("Energy has been successfully corrected.")
									success = True
									i += 1; j += 1
									break
								n_en_try += 1
							if not success: #Если произошёл выход из цикла, значит хотя бы одно из условий выхода было выполнено. При этом успех не был достигнут.
								print("WARNING: acoustics and energy was NOT corrected.")
							break
					else:
						#print("N_ac_try = {}, n_ac_try = {}, filenames_ac_times[j_new+N_ac_try-1] = {}, filenames_ac_times[j_new-1] = {}".format(N_ac_try, n_ac_try, filenames_ac_times[j_new+N_ac_try-1], filenames_ac_times[j_new-1]))
						n_ac_try += 1
				if success:
					break
				else:
					N_ac_try += 1
			if not success:
				print("WARNING: acoustics has NOT been successfully corrected!")
				print(foldername_ac)
				if ext=='.RAW':
					print("filename_ac_j = {}".format("-".join(filenames_ac_info[j])))
				else:
					print("filename_ac_j = {}".format("__".join(filenames_ac_info[j])))
				print("{} files left.".format(len(filenames_ac_info) - j - 1))
				break
		else:
			print("WARNING: suspicious energy time behavior.")
			#delta_j = 1
			time_en_i_minus_1 = np.mean(time_en[i-3:i]) + dt
			if (i+N_try < lc) and (j < len(filenames_ac_info)):
				if abs(time_en[i+N_try-1] - time_en_i_minus_1 - dt*N_try) < EPS_TIME_EN_1:
					print("Energy was not skipped.")
					# i_1, j_1 - cчётчики кадров для энергии и акустики для внутреннего цикла
					for i_1 in range(i,i+N_try-1):
						if j<len(filenames_ac_info):
							calibration[i,0] = energies[i+1]
							calibration[i,1] = filenames_ac_times[j]
							i += 1
							j += 1
						else:
							break
				else:
					count_en_skipped += 1
					n_try = N_try+1
					while (n_try <= 2*N_try) and abs(time_en[i+N_try-1] - time_en_i_minus_1 - dt*n_try) >= EPS_TIME_EN_1:
						n_try += 1
					if abs(time_en[i+N_try-1] - time_en_i_minus_1 - dt*n_try) < EPS_TIME_EN_1:
						print("Energy was skipped {} times".format(n_try-N_try))
						print("Acoustics will be skipped {} times".format(n_try))
						i += N_try - 1
						j += n_try - 1
						count_en_corrected += 1
						print("Energy has been successfully corrected.")
					else:
						print("WARNING: energy was NOT successfully corrected!")
						break

	print("Energy was skipped {0} times, {1} times successfully corrected.".format(count_en_skipped, count_en_corrected))
	if count_en_skipped - count_en_corrected > 0:
		print("WARNING: energy was not corrected {0} times.".format(count_en_skipped - count_en_corrected))
	#print("shift = {}".format(shift))
	return(calibration)

#%% Функция сопоставления акустики и энергии (время рассчитывается от начала выборки (не от предыдущего кадра)).
def compare_not_save_new_method(filename_en, foldername_ac, ext='.bin', shift = None, col_en=9, col_fon=8, col_trig=6, col_times=1, line_length=17, dt=0.1, DELTA = 15, use_trig = True, use_fon = True, old_osc=False):

	#%% Read parameters from the energy file.
	en_param_list = dpb.read_en(filename_en, line_length, col_en, col_fon, col_trig, col_times, use_fon = use_fon)
	if en_param_list == None: #Если массив энергий пуст, завершаем выполнение функции.
		return("Empty.")
	else:
		time_en, energies, i_start, lc = en_param_list
	if use_trig == False:
		print("Trigger column is not used.")
		i_start = 1

	#%% Read list of filenames with luminescence.
	if old_osc:
		filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	else:
		try:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare_new_program(foldername_ac, ext)
		except ValueError:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	if len(filenames_ac_times) == 0: #Если папка с акустикой пуста, сразу завершаем выполнение функции.
		return("Empty")

	#%% Поиск "правильного" сдвига.
	if shift == None:
		shift = shift_search(filenames_ac_times, dt = 0.1, shift_border_min = -3, shift_border_max = 5)
	#shift = 0

	#%% Цикл, сопоставляющий акустику и энергию.
	if shift >= 0:
		time_ac_zero = filenames_ac_times[shift]
		time_en_zero = time_en[i_start]
		shift_i = i_start
		shift_j = shift
	else:
		time_ac_zero = filenames_ac_times[0]
		time_en_zero = time_en[i_start-shift]
		shift_i = i_start-shift
		shift_j = 0

	time_en_new = (np.around((time_en - time_en_zero)*1000)).astype(int) #Времена для энергий в мс, сдвинутые в начало отсчёта.
	time_ac_new = filenames_ac_times - time_ac_zero

	time_en_new = time_en_new[time_en_new >= -is_zero_int]
	time_ac_new = time_ac_new[time_ac_new >= -is_zero_int]

	start_ac_count = 0
	k = 0
	calibration = np.zeros((len(time_en_new), 2))
	for i in range(0,len(time_en_new)):
		for j in range(start_ac_count, len(time_ac_new)):
			#if i==1 and j in range(0,3):
				#print("!!! i={}, j={}, time_en_new[i] = {}, time_ac_new[j]={}, abs={}".format(i, j, time_en_new[i], time_ac_new[j],abs(time_en_new[i]-time_ac_new[j])))
			if abs(time_en_new[i]-time_ac_new[j]) < DELTA:
				#print("i={}, j={}, k={}, time_en_new[i]={}, time_ac_new[j]={}".format(i, j, k, time_en_new[i], time_ac_new[j]))
				calibration[k,0] = energies[i+shift_i]
				calibration[k,1] = filenames_ac_times[j+shift_j]
				start_ac_count = j
				k += 1
				break

	return(calibration)

#%% Функция сопоставления акустики и энергии, если не было пропущено ни одного строба ни на акустике, ни на энергии (без сохранения файлов).
def compare_not_save_no_lost(filename_en, foldername_ac, ext='.bin', shift = None, col_en=9, col_fon=8, col_trig=6, col_times=1, line_length=17, dt=0.1, DELTA = 15, use_trig = True, use_fon = True, old_osc=False):

	#%% Read parameters from the energy file.
	en_param_list = dpb.read_en(filename_en, line_length, col_en, col_fon, col_trig, col_times, use_fon = use_fon)
	if en_param_list == None: #Если массив энергий пуст, завершаем выполнение функции.
		return("Empty.")
	else:
		time_en, energies, i_start, lc = en_param_list
	if use_trig == False:
		print("Trigger column is not used.")
		i_start = 1

	#%% Read list of filenames with luminescence.
	if old_osc:
		filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	else:
		try:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare_new_program(foldername_ac, ext)
		except ValueError:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	if len(filenames_ac_times) == 0: #Если папка с акустикой пуста, сразу завершаем выполнение функции.
		return("Empty")

	#%% Поиск "правильного" сдвига.
	if shift == None:
		shift = shift_search(filenames_ac_times, dt = 0.1, shift_border_min = -3, shift_border_max = 5)
	#shift = 0

	#%% Цикл, сопоставляющий акустику и энергию.
	if shift >= 0:
		time_ac_zero = filenames_ac_times[shift]
		time_en_zero = time_en[i_start]
		shift_i = i_start
		shift_j = shift
	else:
		time_ac_zero = filenames_ac_times[0]
		time_en_zero = time_en[i_start-shift]
		shift_i = i_start-shift
		shift_j = 0

	time_en_new = (np.around((time_en - time_en_zero)*1000)).astype(int) #Времена для энергий в мс, сдвинутые в начало отсчёта.
	time_ac_new = filenames_ac_times - time_ac_zero

	time_en_new = time_en_new[time_en_new >= -is_zero_int]
	time_ac_new = time_ac_new[time_ac_new >= -is_zero_int]

	calibration = np.zeros((min(len(time_en_new), len(time_ac_new)), 2))
	for i in range(0, np.shape(calibration)[0]):
		calibration[i,0] = energies[i+shift_i]
		calibration[i,1] = filenames_ac_times[i+shift_j]

	return(calibration)

def compare_no_lost(filename_en, foldername_ac, folder_to_write, ext='.bin', shift = None, col_en=9, col_fon=8, col_trig=6, col_times=1, line_length=17, dt=0.1, DELTA = 15, use_trig = True, use_fon = True, old_osc=False):


	#%% Read parameters from the energy file.
	en_param_list = dpb.read_en(filename_en, line_length, col_en, col_fon, col_trig, col_times, use_fon = use_fon)
	if en_param_list == None: #Если массив энергий пуст, завершаем выполнение функции.
		return("Empty.")
	else:
		time_en, energies, i_start, lc = en_param_list
	if use_trig == False:
		print("Trigger column is not used.")
		i_start = 1

	#%% Read list of filenames with luminescence.
	if old_osc:
		filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	else:
		try:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare_new_program(foldername_ac, ext)
		except ValueError:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	if len(filenames_ac_times) == 0: #Если папка с акустикой пуста, сразу завершаем выполнение функции.
		return("Empty")

	#%% Поиск "правильного" сдвига.
	if shift == None:
		shift = shift_search(filenames_ac_times, dt = 0.1, shift_border_min = -3, shift_border_max = 5)
	#shift = 0

	#%% Цикл, сопоставляющий акустику и энергию.
	if shift >= 0:
		time_ac_zero = filenames_ac_times[shift]
		time_en_zero = time_en[i_start]
		shift_i = i_start
		shift_j = shift
	else:
		time_ac_zero = filenames_ac_times[0]
		time_en_zero = time_en[i_start-shift]
		shift_i = i_start-shift
		shift_j = 0

	time_en_new = (np.around((time_en - time_en_zero)*1000)).astype(int) #Времена для энергий в мс, сдвинутые в начало отсчёта.
	time_ac_new = filenames_ac_times - time_ac_zero

	time_en_new = time_en_new[time_en_new > -is_zero_int]
	time_ac_new = time_ac_new[time_ac_new > -is_zero_int]

	for i in range(0, min(len(time_en_new), len(time_ac_new))):
		if ext=='.RAW':
			filename_ac_new = os.path.join(folder_to_write, str(energies[i+shift_i]) + "_" + str(i+shift_i) + "_" + "-".join(filenames_ac_info[i+shift_j]) + ext)
			filename_ac = os.path.join(foldername_ac, "-".join(filenames_ac_info[i+shift_j]) + ext)
		else:
			filename_ac_new = os.path.join(folder_to_write, str(energies[i+shift_i]) + "_" + str(i+shift_i) + "_" + "__".join(filenames_ac_info[i+shift_j]) + ext)
			filename_ac = os.path.join(foldername_ac, "__".join(filenames_ac_info[i+shift_j]) + ext)
		sh.copyfile(filename_ac, filename_ac_new)

	return("OK")

def compare_not_save_same_computer(filename_en, foldername_ac, ext='.bin', col_en=9, col_fon=8, col_trig=6, col_times=1, line_length=17, dt=0.1, DELTA = 15, use_fon = True, old_osc=False):

	#%% Read parameters from the energy file.
	en_param_list = dpb.read_en(filename_en, line_length, col_en, col_fon, col_trig, col_times, use_fon = use_fon)
	if en_param_list == None: #Если массив энергий пуст, завершаем выполнение функции.
		return("Empty.")
	else:
		time_en, energies, i_start, lc = en_param_list

	#%% Read list of filenames with luminescence.
	if old_osc:
		filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	else:
		try:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare_new_program(foldername_ac, ext)
		except ValueError:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	if len(filenames_ac_times) == 0: #Если папка с акустикой пуста, сразу завершаем выполнение функции.
		return("Empty")

	time_en = time_en*1000.0
	calibration = np.zeros((min(len(time_en), len(filenames_ac_times)), 2))
	k = 0
	for i in range(0, len(time_en)):
		for j in range(0, len(filenames_ac_times)):
			if abs(time_en[i] - filenames_ac_times[j]) < DELTA:
				calibration[k,0] = energies[i]
				calibration[k,1] = filenames_ac_times[j]
				k += 1

	return(calibration)

def compare_same_computer(filename_en, foldername_ac, folder_to_write, ext='.bin', col_en=9, col_fon=8, col_trig=6, col_times=1, line_length=17, dt=0.1, DELTA = 15, use_fon = True, old_osc=False):


	#%% Read parameters from the energy file.
	en_param_list = dpb.read_en(filename_en, line_length, col_en, col_fon, col_trig, col_times, use_fon = use_fon)
	if en_param_list == None: #Если массив энергий пуст, завершаем выполнение функции.
		return("Empty.")
	else:
		time_en, energies, i_start, lc = en_param_list

	#%% Read list of filenames with luminescence.
	if old_osc:
		filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	else:
		try:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare_new_program(foldername_ac, ext)
		except ValueError:
			filenames_ac_times, filenames_ac_info = dpb.make_file_list_to_compare(foldername_ac, ext)
	if len(filenames_ac_times) == 0: #Если папка с акустикой пуста, сразу завершаем выполнение функции.
		return("Empty")

	time_en = time_en*1000.0
	for i in range(0, len(time_en)):
		for j in range(0, len(filenames_ac_times)):
			if abs(time_en[i] - filenames_ac_times[j]) < DELTA:
				if ext=='.RAW':
					filename_ac_new = os.path.join(folder_to_write, str(energies[i]) + "_" + str(i) + "_" + "-".join(filenames_ac_info[j]) + ext)
					filename_ac = os.path.join(foldername_ac, "-".join(filenames_ac_info[j]) + ext)
				else:
					filename_ac_new = os.path.join(folder_to_write, str(energies[i]) + "_" + str(i) + "_" + "__".join(filenames_ac_info[j]) + ext)
					filename_ac = os.path.join(foldername_ac, "__".join(filenames_ac_info[j]) + ext)
				sh.copyfile(filename_ac, filename_ac_new)
				break

	return("OK")
