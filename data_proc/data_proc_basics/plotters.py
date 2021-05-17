# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:56:08 2018

@author: Дмитрий
"""

import sys, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplc

mpl.rcParams['agg.path.chunksize'] = 10000 #Enables large file plotting.
mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 20})
mpl.rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}',
			r'\usepackage[english,russian]{babel}',
			r'\usepackage{amsmath}',
			r'\boldmath']
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans Serif']})

### For bar plot. ###
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
#####################

def en_wf_plot(t_array, waveform, filename_to_save, style = 'k', color = 'k', lw='1.5', figsize=(10.5, 9.0), dpi=600, xlabel=r'\textbf{Time, ms}', ylabel=r'\textbf{Amplitude, V}'):
	#Graph plotting
	plt.figure(figsize=figsize, dpi=dpi)
	plt.rcParams['text.latex.preamble'] = [r'\boldmath']
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(t_array, waveform, style, color='k', lw=1.5)
	plt.grid()
	plt.minorticks_on()
	plt.tick_params(axis='x', pad=7)
	plt.tick_params(axis='y', pad=5)
	plt.savefig(filename_to_save, bbox_inches='tight')
	plt.close()

def plot_heat_map(data, filename, v_min, v_max, dpi=300, log=False):
	"""
	Function plots heat maps of data numpy 2d array and save result to filename.png.
	"""
	#%% Constants
	#pair_x = 
	#heat map plotting
	mpl.rcdefaults()
	plt.figure(figsize=(20,5), dpi=dpi) #fig_size
	mpl.rcParams.update({'font.size': 22}) #fontsize
	#plt.xlabel(r'$x$')
	#plt.ylabel(r'$y$')
	plt.grid(False)
	plt.minorticks_on()
	plt.tick_params(axis='x', pad=7)
	plt.tick_params(axis='y', pad=5)

	if log==True:
		plt.imshow(data, cmap='jet', aspect='auto', vmin=v_min, vmax=v_max, norm=mplc.LogNorm())
	else:
		plt.imshow(data, cmap='jet', aspect='auto', vmin=v_min, vmax=v_max)

	plt.colorbar()
	plt.savefig(filename, bbox_inches='tight')

	plt.close()

	return True

def plot_heat_map_latex(data, filename, v_min, v_max, dpi=300, font_size = 22, cm = 'gnuP', log=False, aspect=None, extent=None, interpolation=None, grad_num=256, mode='file'):
	"""
	Function plots heat maps of data numpy 2d array and save result to filename.png. The numbers along the axes are recalculated using 'scale' value.
	
	Parameters:
	-----------------------------
	data : 2d array
		Array of values to plot.
	filename : path/string
		Path to the file to save plot in.
	v_min, v_max : float or int
		Minimum and maximum values of the color bar.
	dpi : int
		Resolution of the plot in dots per inch.
	log : boolean
		Use linear (False) or logarithmic (True) color scale.
	scale : float
		pixels per mm of the image.
	"""

	if cm == 'gnuP':
		colors_l = [(0,0,0), (0,0,1), (0,1,0), (1,1,0), (1,0,0)]
		cm = mplc.LinearSegmentedColormap.from_list('gnuP', colors_l, N=grad_num)
	else:
		cm = 'jet'

	fig, ax = plt.subplots(figsize=(20.0, 10.0), dpi=dpi)

	plt.rcParams['text.latex.preamble'] = [r'\boldmath']
	mpl.rcParams.update({'font.size': font_size}) #fontsize
	#plt.rc('font', family='bold', size=24)
	ax.get_xaxis().set_ticks([]) #unset labels
	ax.get_yaxis().set_ticks([])
	plt.grid(False)

	if not extent:
		extent = (0, data.shape[1], 0, data.shape[0])

	if log==True:
		plt.imshow(data, cmap=cm, vmin=v_min, vmax=v_max, norm=mplc.LogNorm(), aspect=aspect, extent=extent, interpolation=interpolation)
		#plt.plot((100,297), (100, 100), 'k', color='w')
	else:
		plt.imshow(data, cmap=cm, vmin=v_min, vmax=v_max, aspect=aspect, extent=extent, interpolation=interpolation)
	plt.colorbar(pad=0.03)
	
	if mode == 'interactive':
		pass
		#plt.show()
	else:
		plt.savefig(filename, bbox_inches='tight')
		plt.close()

	return fig, ax

def plot_heat_map_bar_latex(data, filename, v_min, v_max, cm='gnuP', dpi=300, font_size = 22, bar_width = 10, bar_length = 500, grad_num = 256, scale=(500/88.7), aspect=None, figsize=(20.0, 10.0), bar_sep=7.0, log=False, mode='file'):
	"""
	Function plots heat maps of data numpy 2d array and save result to filename.png.
	"""
	#heat map plotting

	if cm == 'gnuP':
		colors_l = [(0,0,0), (0,0,1), (0,1,0), (1,1,0), (1,0,0)]
		cm = mplc.LinearSegmentedColormap.from_list('gnuP', colors_l, N=grad_num)
	else:
		cm = 'jet'

	x_phys_length = data.shape[1]*scale #physical length of x-axis
	y_phys_length = data.shape[0]*scale #physical length of y-axis
	
	fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
	#fig, ax = plt.subplots(figsize=(20.0*data.shape[1]/720.0, 10.0*data.shape[0]/480.0), dpi=300)
	
	mpl.rc('text', usetex=True)
	plt.rcParams['text.latex.preamble'] = [r'\boldmath']
	mpl.rcParams.update({'font.size': font_size}) #fontsize
	#ax.set_xlabel(r'$x$, $\mu$\textbf{m}')
	#ax.set_ylabel(r'$y$, $\mu$\textbf{m}')
	#plt.minorticks_on()
	#plt.tick_params(axis='x', pad=7)
	#plt.tick_params(axis='y', pad=5)
	ax.get_xaxis().set_ticks([]) #unset labels
	ax.get_yaxis().set_ticks([])
	plt.grid(False)
	extent = (0, x_phys_length, 0, y_phys_length)
	
	####################################################################################
	### barbarbar!!! ###
	
	if bar_length > 950:
		bar_length_caption = f'{bar_length/1000.0:.0f}'
		units = r'\textbf{mm}\normalsize'
	else:
		bar_length_caption = f'{bar_length:.0f}'
		units = r'$\mu$' + r'\textbf{m}'
	bar_label = r'$' + bar_length_caption + r'$\,' + units
	scalebar = AnchoredSizeBar(ax.transData, bar_length, bar_label, 'lower right', pad=0.5, sep=bar_sep, color='white', frameon=False, size_vertical=bar_width, label_top=True)
	ax.add_artist(scalebar)
	
	####################################################################################

	if log==True:
		im = ax.imshow(data, cmap=cm, vmin=v_min, vmax=v_max, norm=mplc.LogNorm(), extent=extent, aspect=aspect)
		#plt.plot((100,297), (100, 100), 'k', color='w')
	else:
		im = ax.imshow(data, cmap=cm, vmin=v_min, vmax=v_max, extent=extent, aspect=aspect)

	fig.colorbar(im, pad=0.03)
	if mode == 'interactive':
		pass
		#plt.show()
	else:
		plt.savefig(filename, bbox_inches='tight')
		plt.close()

	return fig, ax, im

def plot_heat_map_ticks_latex(data, filename, v_min, v_max, scale=(500/90.8), log=False):
	"""
	Function plots heat maps of data numpy 2d array and save result to filename.png.
	"""
	#heat map plotting
	x_phys_length = data.shape[1]*scale #physical length of x-axis
	y_phys_length = data.shape[0]*scale #physical length of y-axis
	
	fig, ax = plt.subplots(figsize=(20.0*data.shape[1]/1920.0, 10*data.shape[0]/1200.0), dpi=300)
	
	plt.rcParams['text.latex.preamble'] = [r'\boldmath']
	mpl.rcParams.update({'font.size': 22}) #fontsize
	#ax.set_xlabel(r'$x$, $\mu$\textbf{m}')
	#ax.set_ylabel(r'$y$, $\mu$\textbf{m}')
	#plt.minorticks_on()
	#plt.tick_params(axis='x', pad=7)
	#plt.tick_params(axis='y', pad=5)
	ax.get_xaxis().set_ticks([]) #unset labels
	ax.get_yaxis().set_ticks([])
	plt.grid(False)
	extent = (0, x_phys_length, 0, y_phys_length)
	
	####################################################################################
	### barbarbar!!! ###
	from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
	import matplotlib.font_manager as fm
	
	bar_label = r'$' + str(bar_length) + r'\,\mu$' + r'\textbf{m}'
	scalebar = AnchoredSizeBar(ax.transData, bar_length, bar_label, 'lower right', pad=0.5, sep=7.0, color='white', frameon=False, size_vertical=0, label_top=True)
	ax.add_artist(scalebar)
	
	####################################################################################

	if log==True:
		im = ax.imshow(data, cmap='jet', vmin=v_min, vmax=v_max, norm=mplc.LogNorm(), extent=extent)
		#plt.plot((100,297), (100, 100), 'k', color='w')
	else:
		im = ax.imshow(data, cmap='jet', vmin=v_min, vmax=v_max, extent=extent)
		
	fig.colorbar(im)
	plt.savefig(filename, bbox_inches='tight')

	plt.close()

	return True

def simple_plotter_from_txt(x, y, filename, ls='ko', xlim = None, ylim = None, dpi=300, font_size = 24, figsize=(10.0, 8.0), log=False, xlabel = None, ylabel = None, mode='file'):
	"""
	Make simple 2D graph.
	"""

	#Включение Latex.
	mpl.rc('text', usetex=True)
	#plt.rc('text.latex', unicode=True)
	plt.rcParams['text.latex.preamble'] = [r'\usepackage[utf8]{inputenc}',
            r'\usepackage[english,russian]{babel}',
            r'\usepackage{amsmath}',
			r'\boldmath']
	plt.rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans Serif']})
	mpl.rcParams.update({'font.size': font_size}) #fontsize
	plt.rc('font', weight='bold')
	
	fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
	
	plt.xticks(fontsize=font_size)
	plt.yticks(fontsize=font_size)
	
	ax.plot(x, y, ls)
	
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.grid()
	plt.minorticks_on()
	plt.tick_params(axis='x', pad=10)
	plt.tick_params(axis='y', pad=10)
	
	if xlim:
		plt.xlim(xlim)
	if ylim:
		plt.xlim(ylim)
	
	if mode == 'interactive':
		pass
	else:
		plt.savefig(filename, bbox_inches='tight')
		plt.close()

	return fig, ax