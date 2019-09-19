#!/usr/bin/env python
# coding: utf-8

import miniscope_file
from load_args import *

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
import scipy.io as sio
import yaml

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params

session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, animal_name)

"""# Prepare data"""
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False, ignore_preexisting=True)

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
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=3, swap_dim=False)

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

opts = params.CNMFParams(params_dict={
    'fnames': memmap_fpath,
    'fr': frate,
    'decay_time': 0.4,
    'pw_rigid': pw_rigid,
    'max_shifts': (5, 5),  # maximum allowed rigid shift
    'gSig_filt': (3, 3),  # size of filter
    'strides': (48, 48),  # start a new patch for pw-rigid motion correction every x pixels
    'overlaps': (24, 24),  # overlap between pathes (size of patch strides+overlaps)
    'max_deviation_rigid': 3,  # maximum deviation allowed for patch with respect to rigid shifts
    'border_nan': border_nan,
    'dims': dims,
    'method_init': 'corr_pnr',  # use this for 1 photon
    'K': None,  # upper bound on number of components per patch, in general None for 1p data
    'gSig': (3, 3),  # gaussian width of a 2D gaussian kernel, which approximates a neuron
    'gSiz': (13, 13),  # average diameter of a neuron, in general 4*gSig+1,
    'merge_thr': 0.7,  # merging threshold, max correlation allowed
    'p': 1,  # order of the autoregressive system
    'tsub': 2,  # downsampling factor in time for initialization, increase if you have memory problems,
    'ssub': 1,  # downsampling factor in space for initialization, increase if you have memory problems
    'rf': 40,  # half-size of the patches in pixels
    'stride': 20,  # amount of overlap between the patches in pixels
                   # (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    'only_init': True,  # set it to True to run CNMF-E
    'nb': gnb,
    'nb_patch': 0,  # number of background components (rank) per patch if gnb>0, else it is set automatically
    'method_deconvolution': 'oasis',  # could use 'cvxpy' alternatively
    'low_rank_background': None,  # None leaves background of each patch intact, True performs global low-rank approximation if gnb>0,
    'update_background_components': True,  # sometimes setting to False improve the results
    'min_corr': 0.8,  # min peak value from correlation image
    'min_pnr': 8,  # min peak to noise ration from PNR image
    'normalize_init': False,  # just leave as is
    'center_psf': True,  # leave as is for 1 photon
    'ssub_B': 2,  # additional downsampling factor in space for background
    'ring_size_factor': 1.4,  # radius of ring is gSiz*ring_size_factor
    'del_duplicates': True,  # whether to remove duplicates from initialization
    'border_pix': 0})  # number of pixels to not consider in the borders)

analysis_start = time.time()
cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
cnm.fit(images)

end = time.time()
print(end - analysis_start)

# ## Component Evaluation
# The components are evaluated in three ways:
# - the shape of each component must be correlated with the data
# - a minimum peak SNR is required over the length of a transient
# - each shape passes a CNN based classifier

min_SNR = 3  # adaptive way to set threshold on the transient size
r_values_min = 0.8  # threshold on space consistency (if you lower more components be accepted, potentially with
# worst quality)
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

print(' ***** ')
print('Number of total components: ', len(cnm.estimates.C))
print('Number of accepted components: ', len(cnm.estimates.idx_components))

# ## Plot results
neuronsToPlot = 20
DeconvTraces = cnm.estimates.S
RawTraces = cnm.estimates.C
SFP = cnm.estimates.A
SFP_dims = list(dims)
SFP_dims.append(SFP.shape[1])
print('Spatial foootprints dimensions (height x width x neurons): ' + str(SFP_dims))

numNeurons = SFP_dims[2]

SFP = np.reshape(SFP.toarray(), SFP_dims, order='F')

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
plt.imshow(np.amax(SFP, axis=2))
plt.colorbar()
plt.title('Spatial footprints')

plt.subplot(2, 2, 2)
plt.figure
plt.title('Example traces (first 50 cells)')
plot_gain = 10  # To change the value gain of traces
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

# ## Register the timestamps for analysis
with open(result_data_dir + '/session_info.yaml', 'r') as f:
    session_info = yaml.load(f, Loader=yaml.FullLoader)
mstime = np.array([], dtype=np.int)

i = 0
for dat_file in session_info['dat_files']:
    with open(dat_file) as f:
        camNum, frameNum, sysClock, buffer = np.loadtxt(f, dtype='float', comments='#', skiprows=1, unpack=True)
    camNumber = camNum[0]
    mstime_idx = np.where(camNum == camNumber)
    this_mstime = sysClock[mstime_idx]
    this_mstime = this_mstime[0:session_info['session_lengths'][i]]
    mstime = np.concatenate([mstime, this_mstime])
    i += 1

mstime[0] = 0

analysis_end = time.time()
analysis_duration = analysis_end - analysis_start
print('Done analyzing. This took a total ' + str(analysis_duration) + ' s')

# ## Save analysis

# In[ ]:


"""# Save the results in HDF5 format"""

save_hdf5 = True
if save_hdf5:
    cnm.save(result_data_dir + '/analysis_results.hdf5')


def find_centroids(SFP):
    centroids = []
    for cell in range(SFP.shape[2]):
        footprint = SFP[:,:,cell]
        max_val = np.max(footprint)
        x, y = np.where(footprint > max_val / 3)
        centroids.append((int(np.median(x)), int(np.median(y))))
    return centroids


meanFrame = np.mean(images[::100], axis=0)
"""# Save the results in Matlab format"""
save_mat = True
if save_mat:
    from scipy.io import savemat

    results_dict = {
        # 'dirName': path_to_analyze,
        'Experiment': animal_name,
        'numFiles': 1,
        'framesNum': len(RawTraces[1]),
        'maxFramesPerFile': 1000,
        'height': dims[0],
        'width': dims[1],
        'camNumber': camNumber,
        'time': mstime,
        'sessionLengths': session_info['session_lengths'],
        # 'analysis_time': analysis_time,
        'meanFrame': meanFrame,
        'Centroids': find_centroids(SFP),
        'CorrProj': cn_filter,
        'PeakToNoiseProj': pnr,
        'RawTraces': RawTraces.conj().transpose(),  # swap time x neurons dimensions
        # 'FiltTraces': cnm.estimates.F_dff,
        'DeconvTraces': cnm.estimates.S.conj().transpose(),
        'SFPs': SFP,
        'numNeurons': SFP_dims[2],
        # 'analysis_duration': analysis_duration
    }

    SFPperm = np.transpose(SFP, [2, 0, 1])
    sio.savemat(result_data_dir + '/SFP.mat', {'SFP': SFPperm})
    sio.savemat(result_data_dir + '/ms.mat', {'ms': results_dict})

# Stop the cluster
cm.stop_server(dview=dview)
