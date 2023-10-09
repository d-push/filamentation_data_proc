# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:56:08 2018

@author: Дмитрий
"""

import sys, glob, os
import numpy as np
import matplotlib as mpl
mpl.rcdefaults()
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from copy import copy
import gc

sys.path.append(r'/usr/bin/latex')

mpl.rcParams['agg.path.chunksize'] = 10000 #Enables large file plotting.
mpl.rc('text', usetex=False)
mpl.rcParams.update({'font.size': 20})
mpl.rcParams['text.latex.preamble'] = r'\usepackage[utf8]{inputenc}, \usepackage[english,russian]{babel}, \usepackage{amsmath}, \boldmath'
mpl.rc('font',**{'family':'sans-serif','sans-serif':['CMU Serif']})

### For bar plot. ###
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
#####################

###Loading colormaps.###
print('Loading colormaps...')
path_mask_for_colormaps = os.path.join(os.getcwd(), 'COLORMAPS', '*.ini')
filenames = glob.glob(path_mask_for_colormaps)
cm_dict = {}
for filename in filenames:
	key = os.path.basename(filename).split('.ini')[0]
	values = np.loadtxt(filename)
	cm_dict[key] = values
print('OK.')

def im_to_dat_convert(im_coord, size_x):
	'''Converts image coordinates to data coordinates.'''
	
	dat_coord = np.zeros_like(im_coord)
	dat_coord[1] = im_coord[0]
	dat_coord[0] = size_x - im_coord[1]
	
	return(dat_coord)
	
def dat_to_im_convert(dat_coord, size_x):
	'''Converts image coordinates to data coordinates.'''
	
	im_coord = np.zeros_like(dat_coord)
	im_coord[0] = dat_coord[1]
	im_coord[1] = size_x - dat_coord[0]
	
	return(im_coord)
	

def en_wf_plot(t_array, waveform, filename_to_save, style = 'k', color = 'k', lw='1.5', figsize=(10.5, 9.0), dpi=600, xlabel=r'\textbf{Time, ms}', ylabel=r'\textbf{Amplitude, V}'):
	#Graph plotting
	plt.figure(figsize=figsize, dpi=dpi)
	#plt.rcParams['text.latex.preamble'] = [r'\boldmath']
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
		plt.imshow(data, cmap='jet', aspect='auto', norm=mplc.LogNorm(vmin = v_min, vmax = v_max))
	else:
		plt.imshow(data, cmap='jet', aspect='auto', vmin=v_min, vmax=v_max)

	plt.colorbar()
	plt.savefig(filename, bbox_inches='tight')

	plt.close()

	return True

def plot_heat_map_latex(data, filename, v_min, v_max, extent = None, scale=None, dpi=300, font_size = 22, cm = 'gnuP', log=False, aspect=None, interpolation=None, grad_num=256, mode='file', mpl_patches=[], figsize=None, nbins = (5,5), cnumticks = None):
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

	if cm == 'gnuP' or cm is None:
		colors_l = [(0,0,0), (0,0,1), (0,1,0), (1,1,0), (1,0,0)]
		cm = mplc.LinearSegmentedColormap.from_list('gnuP', colors_l, N=grad_num)
	elif cm == 'gnuPw':
		colors_l = [(1,1,1), (0,0,1), (0,1,0), (1,1,0), (1,0,0)]
		cm = mplc.LinearSegmentedColormap.from_list('gnuPw', colors_l, N=grad_num)
	elif cm in cm_dict:
		colors_l = cm_dict[cm]
		cm = mplc.LinearSegmentedColormap.from_list(cm, colors_l, N=grad_num)


	mpl.rc('text', usetex=True)
	mpl.rcParams['text.latex.preamble'] = r"\usepackage[utf8]{inputenc} \usepackage[english,russian]{babel}  \usepackage{amsmath} \boldmath"
	mpl.rcParams.update({'font.size': font_size}) #fontsize
	
	if figsize is None:
		figsize = (20.0, 10.0)
	fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

	if scale is None and extent is None:
		ax.get_xaxis().set_ticks([]) #unset labels
		ax.get_yaxis().set_ticks([])
	plt.grid(False)

	if extent is None:
		if scale is not None and len(scale) == 2:
			extent = (-data.shape[1]*scale[0]/2.0, data.shape[1]*scale[0]/2.0, -data.shape[0]*scale[1]/2.0, data.shape[0]*scale[1]/2.0)
			aspect = 1
			plt.xticks(fontsize=font_size)
			plt.yticks(fontsize=font_size)
		elif scale is not None and len(scale) == 1:
			extent = (-data.shape[1]*scale[0]/2.0, data.shape[1]*scale[0]/2.0, -data.shape[0]*scale[0]/2.0, data.shape[0]*scale[0]/2.0)
		else:
			extent = (0, data.shape[1], 0, data.shape[0])
	
	plt.locator_params(axis='x', nbins=nbins[0])
	plt.locator_params(axis='y', nbins=nbins[1])
	plt.tick_params(axis='x', pad=7)
	plt.tick_params(axis='y', pad=5)
	
	if not cnumticks:
		cticks = None
	else:
		cticks = np.linspace(v_min, v_max, cnumticks)

	if log==True:
		plt.imshow(data, cmap=cm, norm=mplc.LogNorm(vmin=v_min, vmax=v_max), aspect=aspect, extent=extent, interpolation=interpolation)
		#plt.plot((100,297), (100, 100), 'k', color='w')
	else:
		plt.imshow(data, cmap=cm, vmin=v_min, vmax=v_max, aspect=aspect, extent=extent, interpolation=interpolation)
	plt.colorbar(pad=0.03, ticks = cticks)
	
	for patch in mpl_patches:
		patch_c = copy(patch)
		ax.add_patch(patch_c)
			
	if mode == 'interactive':
		pass
		#plt.show()
	else:
		plt.savefig(filename, bbox_inches='tight')
		plt.close()

	return fig, ax

def plot_heat_map_bar_latex(data, filename, v_min, v_max, cm='gnuP', dpi=300, font_size = 22, bar_width = 10, bar_length = 500, grad_num = 256, scale=(500/88.7), aspect=None, figsize=(20.0, 10.0), bar_sep=7.0, bar_font_size = None, log=False, mode='file', cnumticks = None):
	"""
	Function plots heat maps of data numpy 2d array and save result to filename.png.
	"""
	#heat map plotting

	if cm == 'gnuP' or cm is None:
		colors_l = [(0,0,0), (0,0,1), (0,1,0), (1,1,0), (1,0,0)]
		cm = mplc.LinearSegmentedColormap.from_list('gnuP', colors_l, N=grad_num)
	elif cm == 'gnuPw':
		colors_l = [(1,1,1), (0,0,1), (0,1,0), (1,1,0), (1,0,0)]
		cm = mplc.LinearSegmentedColormap.from_list('gnuPw', colors_l, N=grad_num)
	elif cm in cm_dict:
		colors_l = cm_dict[cm]
		cm = mplc.LinearSegmentedColormap.from_list(cm, colors_l, N=grad_num)

	
	if type(scale) == list and len(scale) == 1:
		scale = scale[0]

	x_phys_length = data.shape[1]*scale #physical length of x-axis
	y_phys_length = data.shape[0]*scale #physical length of y-axis
	
	mpl.rc('text', usetex=True)
	mpl.rcParams['text.latex.preamble'] = r"\usepackage[utf8]{inputenc} \usepackage[english,russian]{babel}  \usepackage{amsmath} \boldmath"
	mpl.rc('font', weight='bold', family='sans-serif')
	mpl.rcParams.update({'font.size': font_size}) #fontsize
	
	fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
	#fig, ax = plt.subplots(figsize=(20.0*data.shape[1]/720.0, 10.0*data.shape[0]/480.0), dpi=300)

	ax.get_xaxis().set_ticks([]) #unset labels
	ax.get_yaxis().set_ticks([])
	plt.grid(False)
	extent = (0, x_phys_length, 0, y_phys_length)
	
	####################################################################################
	### barbarbar!!! ###
	
	if bar_length > 950:
		bar_length_caption = f'{bar_length/1000.0:.0f}'
		units = r'\textbf{mm}'
	else:
		bar_length_caption = f'{bar_length:.0f}'
		units = r'$\boldmath{\mu}$' + r'\textbf{m}'
	bar_label = r'$' + bar_length_caption + r'$\,' + units
	scalebar = AnchoredSizeBar(ax.transData, bar_length, bar_label, 'lower right', pad=0.5, sep=bar_sep, color='white', frameon=False, size_vertical=bar_width, label_top=True)
	ax.add_artist(scalebar)
	
	####################################################################################

	if log==True:
		im = ax.imshow(data, cmap=cm, norm=mplc.LogNorm(vmin=v_min, vmax=v_max), extent=extent, aspect=aspect)
		#plt.plot((100,297), (100, 100), 'k', color='w')
	else:
		im = ax.imshow(data, cmap=cm, vmin=v_min, vmax=v_max, extent=extent, aspect=aspect)

	if not cnumticks:
		cticks = None
	else:
		cticks = np.linspace(v_min, v_max, cnumticks)
	fig.colorbar(im, pad=0.03, ticks = cticks)
	
	if mode == 'interactive':
		return fig, ax, im
		#plt.show()
	else:
		plt.savefig(filename, bbox_inches='tight')
		fig.clf()
		plt.close(fig)
		gc.collect()

	return 0

def plot_heat_map_ticks_latex(data, filename, v_min, v_max, scale=(500/90.8), log=False):
	"""
	Function plots heat maps of data numpy 2d array and save result to filename.png.
	"""
	#heat map plotting
	x_phys_length = data.shape[1]*scale #physical length of x-axis
	y_phys_length = data.shape[0]*scale #physical length of y-axis
	
	fig, ax = plt.subplots(figsize=(20.0*data.shape[1]/1920.0, 10*data.shape[0]/1200.0), dpi=300)
	
	plt.rcParams['text.latex.preamble'] = r'\boldmath'
	mpl.rc('font', weight='bold', family='sans-serif')
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
		im = ax.imshow(data, cmap='jet', norm=mplc.LogNorm(vmin=v_min, vmax=v_max), extent=extent)
		#plt.plot((100,297), (100, 100), 'k', color='w')
	else:
		im = ax.imshow(data, cmap='jet', vmin=v_min, vmax=v_max, extent=extent)
		
	fig.colorbar(im)
	plt.savefig(filename, bbox_inches='tight')

	plt.close()
	fig.clf()

	return True

def simple_plotter_from_txt(x, y, filename, ls='ko', xlim = None, ylim = None, dpi=300, font_size = 24, figsize=(10.0, 8.0), log=False, xlabel = None, ylabel = None, yscale_right = False, mode='file'):
	"""
	Make simple 2D graph.
	"""

	#Включение Latex.
	mpl.rcdefaults()
	mpl.rc('text', usetex=False)
	plt.rcParams['text.latex.preamble'] = r"\usepackage[utf8]{inputenc} \usepackage[english,russian]{babel}  \usepackage{amsmath} \boldmath"
	mpl.rc('font', weight='bold', family='serif')
	mpl.rcParams.update({'font.size': font_size}) #fontsize
	csfont = {'fontname':'CMU Serif', 'family' : 'serif'}
	
	fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
	
	plt.xticks(fontsize=font_size)
	plt.yticks(fontsize=font_size)
	
	ax.plot(x, y, ls)
	
	plt.xlabel(xlabel, **csfont, labelpad=1)
	plt.grid()
	plt.minorticks_on()
	plt.tick_params(axis='x', pad=10)
	plt.tick_params(axis='y', pad=10)
	plt.locator_params(axis='y', nbins=6)
	
	if xlim:
		plt.xlim(xlim)
	if ylim:
		plt.xlim(ylim)
		
	if yscale_right:
		ax.yaxis.tick_right()
		ax.yaxis.set_label_position("right")
	plt.ylabel(ylabel, **csfont, labelpad=10)
	
	if log:
		plt.yscale("log")
	
	if mode == 'interactive':
		pass
	else:
		plt.savefig(filename, bbox_inches='tight')
		plt.close()

	return fig, ax

#%% Plotters for M^2 programm.
def simple_image_plotter(filename_to_plot, data):
	'''
	Simple modes plotter
	'''
	
	mpl.rcdefaults()
	fig, ax = plt.subplots()
	plt.imshow(data, aspect='equal')
	plt.savefig(filename_to_plot, bbox_inches = 'tight')
	plt.close()
	
	return(0)

def plot_sigma_z(filename_to_plot, z, sigma_list, err_sigma_list, ellipticity_list=None, err_ellipticity_list = None, dim=r'mm', xlim = None, ylim = None, ylim1 = None, dpi=300, font_size = 22, figsize=(10.0, 8.0), log=False, xlabel = r'$z$', ylabel = r'$\sigma$, mm', ylabel1 = r'Ellipticity, arb.un.', latex=True, bold=True):
	'''
	Plot sigma(z) dependence.
	'''
	
	#Constants
	fmt_list = ['ko', 'rs']
	fmt_list1 = ['r^-']

	#Включение Latex.
	mpl.rcdefaults()
	if latex:
		mpl.rc('text', usetex=True)
		mpl.rcParams.update({'text.latex.preamble' : r'\usepackage{amsmath} \boldmath'})
		if bold:
			mpl.rc('font', weight='bold', family='serif')
	mpl.rcParams.update({'font.size': font_size})
	
	csfont = {'fontname':'CMU Serif', 'family' : 'serif'}
	#csfont = {'fontname':'Comic Sans MS'}
	
	fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
	for i,sigma in enumerate(sigma_list):
		ax1.errorbar(z, sigma, yerr = err_sigma_list[i], fmt=fmt_list[i], capsize=5)
	
	if ellipticity_list is not None:
		ax2 = ax1.twinx()
		for i,ellipticity in enumerate(ellipticity_list):
			ax2.errorbar(z, ellipticity_list[i], yerr = err_ellipticity_list[i], fmt=fmt_list1[i], capsize=5)
		
	plt.xticks()
	plt.yticks()
	
	if latex:
		if bold:
			if dim == '':
				xlabel = r'\textbf{' + xlabel + r'}'
			else:
				xlabel = r'\textbf{' + xlabel + r', '+dim+r'}'
			ylabel = r'\textbf{' + ylabel + r'}'
			if ellipticity_list is not None:
				ylabel1 = r'\textbf{' + ylabel1+ r'}'
		else:
			if dim !='':
				xlabel = xlabel+r', \rmseries{'+dim+r'}'
			ylabel = r'\rmseries{' + ylabel + r'}'
	else:
		xlabel = xlabel+', '+dim
	plt.xlabel(xlabel, fontsize=24, **csfont, labelpad=2)
	ax1.set_ylabel(ylabel, fontsize=24, **csfont, labelpad=7)
	if ellipticity_list is not None:
		ax2.set_ylabel(ylabel1, fontsize=24, **csfont, labelpad=7)
	plt.grid()
	plt.minorticks_on()
	plt.tick_params(axis='x', pad=7)
	plt.tick_params(axis='y', pad=7)
	
	if xlim is not None:
		plt.xlim(xlim)
	if ylim is not None:
		ax1.set_ylim(ylim)
	if ylim1 is not None:
		ax2.set_ylim(ylim1)
		
	print(f'xlabel = {xlabel}, ylabel = {ylabel}')
	
	plt.savefig(filename_to_plot, bbox_inches='tight')
	plt.close()
	
	return(0)
