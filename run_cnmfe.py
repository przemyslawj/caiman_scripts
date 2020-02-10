#!/usr/bin/env python
# coding: utf-8

import miniscope_file
import results_format
from load_args import *

import logging
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
import yaml
import sys

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import inspect_correlation_pnr


logging.basicConfig(level=logging.INFO)
session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, animal_name)

"""# Prepare data"""
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=min(3, ncores), single_thread=False, ignore_preexisting=True)

# ## Load Motion Corrected data
memmap_fpath = miniscope_file.get_joined_memmap_fpath(result_data_dir)

Yr, dims, T = cm.load_memmap(memmap_fpath)
images = Yr.T.reshape((T,) + dims, order='F')
logging.info('Loaded memmap file')

# ## Read CNMFE params
local_params_fpath = '/'.join([
    local_rootdir,
    downsample_subpath,
    experiment_month,
    'cnmfe_params.csv'])

if os.path.isfile(local_params_fpath):
    import pandas as pd
    params_csv = pd.read_csv(local_params_fpath)
    animal_params = params_csv[params_csv['animal'] == animal_name]
else:
    animal_params = []
if len(animal_params) > 0:
    logging.info('Using CNMFE params file')
    gSig = (animal_params['gSig'].values[0], animal_params['gSig'].values[0])
    gSiz = (animal_params['gSiz'].values[0], animal_params['gSiz'].values[0])
    min_corr = animal_params['min_corr'].values[0]
    min_pnr = animal_params['min_pnr'].values[0]
    ring_size_factor = 2.0
else:
    logging.info('Using default CNMFE params')
    # ## Setup CNMFE params
    # gSig = (3, 3)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSig = (4, 4)
    # gSiz = (13, 13)     # average diameter of a neuron, in general 4*gSig+1
    gSiz = (17, 17)  # average diameter of a neuron, in general 4*gSig+1
    min_corr = .8  # min peak value from correlation image
    #min_corr = .7
    min_pnr = 8  # min peak to noise ration from PNR image
    ring_size_factor = 1.6  # radius of ring is gSiz*ring_size_factor


frate = 20
pw_rigid = False  # flag for pw-rigid motion correction
border_nan = 'copy'
Ain = None  # possibility to seed with predetermined binary masks
gnb = 0  # number of background components (rank) if positive, else exact ring model with following settings
# gnb= 0: Return background as b and W
# gnb=-1: Return full rank background B
# gnb<-1: Don't return background
decay_time = 0.4  # length of a typical transient in seconds

# Compute some summary images (correlation and peak to noise) while downsampling temporally 5x to speedup the process and avoid memory overflow
# change swap dim if output looks weird, it is a problem with tiffile
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::20], gSig=gSig[0], swap_dim=False)
if hasattr(sys, 'ps1'): # interactive mode
    inspect_correlation_pnr(cn_filter, pnr)
    plt.show(block=True)

p = 1               # order of the autoregressive system
K = None            # upper bound on number of components per patch, in general None for 1p data
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
ssub_B = 2          # additional downsampling factor in space for background


opts = params.CNMFParams(params_dict={'dims': dims,
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
                                'border_pix': 2})                # number of pixels to not consider in the borders)

logging.info('Starting CNMF')
analysis_start = time.time()
cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
cnm.fit(images)

end = time.time()
logging.info(end - analysis_start)
logging.info('Finished CNMF')

# ## Component Evaluation
# The components are evaluated in three ways:
# - the shape of each component must be correlated with the data
# - a minimum peak SNR is required over the length of a transient
# - each shape passes a CNN based classifier

min_SNR = 3  # adaptive way to set threshold on the transient size
r_values_min = 0.8  # threshold on space consistency (if you lower more components be accepted, potentially with
# worst quality)
logging.info(' ***** ')
logging.info('Number of total components: %d', len(cnm.estimates.C))
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

logging.info('Number of accepted components: %d', len(cnm.estimates.idx_components))

# ## Plot results
neuronsToPlot = 20
RawTraces = cnm.estimates.C
maxRawTraces = np.amax(RawTraces)

plt.figure(figsize=(30, 15))
plt.subplot(3, 2, 1)
plt.imshow(cn_filter)
plt.colorbar()
plt.title('Correlation projection')
plt.subplot(3, 2, 3)
plt.imshow(pnr)
plt.colorbar()
plt.title('PNR')
plt.subplot(3, 2, 5)
plt.imshow(np.amax(results_format.readSFP(cnm), axis=2))
plt.colorbar()
plt.title('Spatial footprints')

plt.subplot(3, 2, 2)
plt.figure
plt.title('Example traces (first 50 cells)')
plot_gain = 10  # To change the value gain of traces
plot_maxlen_sec = 200
numNeurons = cnm.estimates.A.shape[1]
if numNeurons >= neuronsToPlot:
    for i in range(neuronsToPlot):
        if i == 0:
            plt.plot(RawTraces[i, 0:plot_maxlen_sec*frate], 'k')
        else:
            trace = RawTraces[i, 0:plot_maxlen_sec*frate] + maxRawTraces * i / plot_gain
            plt.plot(trace, 'k')
else:
    for i in range(numNeurons):
        if i == 0:
            plt.plot(RawTraces[i, :], 'k')
        else:
            trace = RawTraces[i, :] + maxRawTraces * i / plot_gain
            plt.plot(trace, 'k')

plt.subplot(3, 2, 4)
plt.figure
plt.title('Deconvolved traces (first 50 cells)')
DeconvTraces = cnm.estimates.S
plot_gain = 20  # To change the value gain of traces
if numNeurons >= neuronsToPlot:
    for i in range(neuronsToPlot):
        if i == 0:
            plt.plot(DeconvTraces[i, 0:plot_maxlen_sec*frate], 'k')
        else:
            trace = DeconvTraces[i, 0:plot_maxlen_sec*frate] + maxRawTraces * i / plot_gain
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
# workaround for issue: https://github.com/flatironinstitute/CaImAn/issues/673
cnm.estimates.r_values = np.where(np.isnan(cnm.estimates.r_values), -1, cnm.estimates.r_values)
cnm.estimates.SNR_comp = np.where(np.isnan(cnm.estimates.SNR_comp), 0, cnm.estimates.SNR_comp)

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
    mstime, camNumber = results_format.concat_session_timestamps(session_info, local_rootdir,
                                                                 downsample_subpath, rclone_config)
    results_format.save_matlab(cnm, session_info, result_data_dir, images[::100],
                               mstime, camNumber)

# Stop the cluster
cm.stop_server(dview=dview)
