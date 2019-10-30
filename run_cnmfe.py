#!/usr/bin/env python
# coding: utf-8

import miniscope_file
from load_args import *

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
import yaml
import logging

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params

import results_format


logging.basicConfig(level=logging.INFO)
session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, animal_name)

"""# Prepare data"""
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=min(3, ncores), single_thread=False, ignore_preexisting=True)

# ## Load Motion Corrected data
load_mmap = True
if not load_mmap:
    mc_fpath = result_data_dir + '/mc.avi'
    memmap_fpath = cm.save_memmap([mc_fpath], base_name='memmap_',
                                  order='C', border_to_0=0, dview=dview)
else:
    memmap_fpath = miniscope_file.get_joined_memmap_fpath(result_data_dir)

Yr, dims, T = cm.load_memmap(memmap_fpath)
images = Yr.T.reshape((T,) + dims, order='F')
print('Loaded memmap file')

# Compute some summary images (correlation and peak to noise) while downsampling temporally 5x to speedup the process and avoid memory overflow
# change swap dim if output looks weird, it is a problem with tiffile
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::20], gSig=3, swap_dim=False)

# Plot the results of the correlation/PNR projection
plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.imshow(cn_filter)
plt.colorbar()
plt.title('Correlation projection')
plt.subplot(2, 2, 2)
plt.imshow(pnr)
plt.colorbar()
plt.title('PNR')
plt.savefig(result_data_dir + '/' + 'pnr.svg', edgecolor='w', format='svg', transparent=True)

# ## Run CNMFE
frate = 20
pw_rigid = False  # flag for pw-rigid motion correction
border_nan = 'copy'
Ain = None  # possibility to seed with predetermined binary masks
gnb = 0  # number of background components (rank) if positive, else exact ring model with following settings
# gnb= 0: Return background as b and W
# gnb=-1: Return full rank background B
# gnb<-1: Don't return background
decay_time = 0.4  # length of a typical transient in seconds

# motion correction parameters
motion_correct = False  # flag for motion correction
pw_rigid = False  # flag for pw-rigid motion correction

gSig_filt = (3, 3)  # size of filter, in general gSig (see below),
#                      change this one if algorithm does not work
max_shifts = (5, 5)  # maximum allowed rigid shift
strides = (48, 48)  # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3
border_nan = 'copy'

mc_dict = {
    'fnames': memmap_fpath,
    'fr': frate,
    'decay_time': decay_time,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'gSig_filt': gSig_filt,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': border_nan
}

opts = params.CNMFParams(params_dict=mc_dict)

# opts = params.CNMFParams(params_dict={
#     'memory_fact': 0.8,
#     'fnames': memmap_fpath,
#     'fr': frate,
#     'decay_time': 0.4,
#     'pw_rigid': pw_rigid,
#     'max_shifts': (5, 5),  # maximum allowed rigid shift
#     'gSig_filt': (3, 3),  # size of filter
#     'strides': (48, 48),  # start a new patch for pw-rigid motion correction every x pixels
#     'overlaps': (24, 24),  # overlap between pathes (size of patch strides+overlaps)
#     'max_deviation_rigid': 3,  # maximum deviation allowed for patch with respect to rigid shifts
#     'border_nan': border_nan
# })
# opts.change_params(params_dict={
#     'dims': dims,
#     'method_init': 'corr_pnr',  # use this for 1 photon
#     'K': None,  # upper bound on number of components per patch, in general None for 1p data
#     'gSig': (3, 3),  # gaussian width of a 2D gaussian kernel, which approximates a neuron
#     'gSiz': (13, 13),  # average diameter of a neuron, in general 4*gSig+1,
#     'merge_thr': 0.7,  # merging threshold, max correlation allowed
#     'p': 1,  # order of the autoregressive system
#     'tsub': 2,  # downsampling factor in time for initialization, increase if you have memory problems,
#     'ssub': 1,  # downsampling factor in space for initialization, increase if you have memory problems
#     'rf': 40,  # half-size of the patches in pixels
#     'stride': 20,  # amount of overlap between the patches in pixels
#                    # (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
#     'only_init': True,  # set it to True to run CNMF-E
#     'nb': gnb,
#     'nb_patch': 0,  # number of background components (rank) per patch if gnb>0, else it is set automatically
#     'method_deconvolution': 'oasis',  # could use 'cvxpy' alternatively
#     'low_rank_background': None,  # None leaves background of each patch intact, True performs global low-rank approximation if gnb>0,
#     'update_background_components': True,  # sometimes setting to False improve the results
#     'min_corr': 0.8,  # min peak value from correlation image
#     'min_pnr': 8,  # min peak to noise ration from PNR image
#     'normalize_init': False,  # just leave as is
#     'center_psf': True,  # leave as is for 1 photon
#     'ssub_B': 2,  # additional downsampling factor in space for background
#     'ring_size_factor': 1.4,  # radius of ring is gSiz*ring_size_factor
#     'del_duplicates': True,  # whether to remove duplicates from initialization
#     'border_pix': 0})  # number of pixels to not consider in the borders)
p = 1               # order of the autoregressive system
K = None            # upper bound on number of components per patch, in general None for 1p data
gSig = (3, 3)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = (13, 13)     # average diameter of a neuron, in general 4*gSig+1

Ain = None          # possibility to seed with predetermined binary masks
merge_thr = .7      # merging threshold, max correlation allowed
rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 20    # amount of overlap between the patches in pixels
#                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 2            # downsampling factor in time for initialization,
#                     increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization,
#                     increase if you have memory problems
#                     you can pass them here as boolean vectors
low_rank_background = None  # None leaves background of each patch intact,
#                     True performs global low-rank approximation if gnb>0
gnb = 0             # number of background components (rank) if positive,
#                     else exact ring model with following settings
#                         gnb= 0: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb<-1: Don't return background
nb_patch = 0        # number of background components (rank) per patch if gnb>0,
#                     else it is set automatically
min_corr = .75     # min peak value from correlation image
min_pnr = 8        # min peak to noise ration from PNR image
ssub_B = 2          # additional downsampling factor in space for background
ring_size_factor = 1.6  # radius of ring is gSiz*ring_size_factor

opts.change_params(params_dict={'dims': dims,
                                'method_init': 'corr_pnr',  # use this for 1 photon
                                'K': K,
                                'gSig': gSig,
                                'gSiz': gSiz,
                                'merge_thr': merge_thr,
                                'p': p,
                                'tsub': tsub,
                                'ssub': ssub,
                                'rf': rf,
                                'stride': stride_cnmf,
                                'only_init': True,    # set it to True to run CNMF-E
                                'nb': gnb,
                                'nb_patch': nb_patch,
                                'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                'low_rank_background': low_rank_background,
                                'update_background_components': True,  # sometimes setting to False improve the results
                                'min_corr': min_corr,
                                'min_pnr': min_pnr,
                                'normalize_init': False,               # just leave as is
                                'center_psf': True,                    # leave as is for 1 photon
                                'ssub_B': ssub_B,
                                'ring_size_factor': ring_size_factor,
                                'del_duplicates': True,                # whether to remove duplicates from initialization
                                'border_pix': 0})                # number of pixels to not consider in the borders)

print('Starting CNMF')
analysis_start = time.time()
cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
cnm.fit(images)

end = time.time()
print(end - analysis_start)
print('Finished CNMF')

# ## Component Evaluation
# The components are evaluated in three ways:
# - the shape of each component must be correlated with the data
# - a minimum peak SNR is required over the length of a transient
# - each shape passes a CNN based classifier

min_SNR = 3  # adaptive way to set threshold on the transient size
r_values_min = 0.8  # threshold on space consistency (if you lower more components be accepted, potentially with
# worst quality)
print(' ***** ')
print('Number of total components: ', len(cnm.estimates.C))
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

print('Number of accepted components: ', len(cnm.estimates.idx_components))

# ## Plot results
RawTraces = cnm.estimates.C
neuronsToPlot = 20
maxRawTraces = np.amax(RawTraces)

plt.figure(figsize=(30, 15))
plt.subplot(3, 4, 9)
plt.subplot(3, 4, 2)
plt.imshow(cn_filter)
plt.colorbar()
plt.title('Correlation projection')
plt.subplot(3, 4, 6)
plt.imshow(pnr)
plt.colorbar()
plt.title('PNR')
plt.subplot(3, 4, 10)
plt.imshow(np.amax(results_format.readSFP(cnm), axis=2))
plt.colorbar()
plt.title('Spatial footprints')

plt.subplot(2, 2, 2)
plt.figure
plt.title('Example traces (first 50 cells)')
plot_gain = 10  # To change the value gain of traces
numNeurons = cnm.estimates.A.shape[1]
if numNeurons >= neuronsToPlot:
    for i in range(neuronsToPlot):
        if i == 0:
            plt.plot(RawTraces[i, :], 'k')
        else:
            trace = RawTraces[i, :] + maxRawTraces * i / plot_gain
            plt.plot(trace, 'k')
else:
    for i in range(numNeurons):
        if i == 0:
            plt.plot(RawTraces[i, :], 'k')
        else:
            trace = RawTraces[i, :] + maxRawTraces * i / plot_gain
            plt.plot(trace, 'k')

plt.subplot(2, 2, 4)
plt.figure
plt.title('Deconvolved traces (first 50 cells)')
DeconvTraces = cnm.estimates.S
plot_gain = 20  # To change the value gain of traces
if numNeurons >= neuronsToPlot:
    for i in range(neuronsToPlot):
        if i == 0:
            plt.plot(DeconvTraces[i, :], 'k')
        else:
            trace = DeconvTraces[i, :] + maxRawTraces * i / plot_gain
            plt.plot(trace, 'k')
else:
    for i in range(numNeurons):
        if i == 0:
            plt.plot(DeconvTraces[i, :], 'k')
        else:
            trace = DeconvTraces[i, :] + maxRawTraces * i / plot_gain
            plt.plot(trace, 'k')

# Save summary figure
plt.savefig(result_data_dir + '/' + 'summary_figure.svg', edgecolor='w', format='svg', transparent=True)

"""# Save the results in HDF5 format"""

save_hdf5 = True
if save_hdf5:
    cnm.save(result_data_dir + '/analysis_results.hdf5')

analysis_end = time.time()
analysis_duration = analysis_end - analysis_start
print('Done analyzing. This took a total ' + str(analysis_duration) + ' s')

# ## Save analysis
# ## Register the timestamps
with open(result_data_dir + '/session_info.yaml', 'r') as f:
    session_info = yaml.load(f, Loader=yaml.FullLoader)

"""# Save the results in Matlab format"""
save_mat = True
if save_mat:
    results_format.save_matlab(cnm, session_info, result_data_dir, images[::100])

# Stop the cluster
cm.stop_server(dview=dview)
