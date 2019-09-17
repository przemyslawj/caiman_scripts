#!/usr/bin/env python
# coding: utf-8

import miniscope_file

import scipy.io as sio
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
import numpy as np
import pandas as pd

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params

experiment_month = os.environ['EXP_MONTH']
experiment_title = os.environ['EXP_TITLE']
experiment_date = os.environ['EXP_DATE']
animal_name = os.environ['ANIMAL']
spatial_downsampling = int(os.environ['DOWNSAMPLE'])
downsample_subpath = os.environ['DOWNSAMPLE_SUBPATH']
local_rootdir = os.environ['LOCAL_ROOTDIR']

local_miniscope_path = '/'.join([
    local_rootdir,
    downsample_subpath,
    experiment_month,
    experiment_title,
    experiment_date])
session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, animal_name)

"""# Prepare data"""
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False, ignore_preexisting=True)


# ## Load Motion Corrected data
result_data_dir = '/'.join([local_miniscope_path, 'caiman', animal_name])
pd.read_csv(result_data_dir + '/session_info.csv')

load_mmap = True
if not load_mmap:
    mc_fpath = result_data_dir + '/mc.avi'
    fname_new = cm.save_memmap([mc_fpath], base_name='memmap_',
                                order='C', border_to_0=0, dview=dview)
else:
    fname_new = [f for f in os.listdir(result_data_dir) if f.endswith('mmap')][0]

# load memory mappable file
Yr, dims, T = cm.load_memmap(fname_new)
images = Yr.T.reshape((T,) + dims, order='F')

# Compute some summary images (correlation and peak to noise) while downsampling temporally 5x to speedup the process and avoid memory overflow
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=3, swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile

#Plot the results of the correlation/PNR projection
plt.figure(figsize=(20,10))
plt.subplot(2, 2, 1); plt.imshow(cn_filter); plt.colorbar(); plt.title('Correlation projection')
plt.subplot(2, 2, 2); plt.imshow(pnr); plt.colorbar(); plt.title('PNR')
plt.savefig(result_data_dir + '/' + 'pnr.svg', edgecolor='w', format='svg', transparent=True)


# ## Run CNMFE

# In[10]:

frate = 20
opts_dict = {
  'fr': frate,
  'use_cuda' : True,
  'memory_fact': 1.0,
  'decay_time': 0.4,
  'splits_rig': 20,  # for parallelization split the movies in num_splits chunks across time
  'method_init': 'corr_pnr',  # use this for 1 photon
  'K': None, # upper bound on number of components per patch, in general None
  'gSig': [2, 2], # gaussian width of a 2D gaussian kernel, which approximates a neuron
  'gSiz': [9, 9], # average diameter of a neuron, in general 4*gSig+1
  #'gSig': [3, 3], # gaussian width of a 2D gaussian kernel, which approximates a neuron
  #'gSiz': [15, 15], # average diameter of a neuron, in general 4*gSig+1
  'merge_thr': 0.65, # merging threshold, max correlation allowed
  'p': 1, # order of the autoregressive system
  'tsub': 2, # downsampling factor in time for initialization
  'ssub': 2, # downsampling factor in space for initialization
  'rf': 40, # half-size of the patches in pixels
  'stride': 20, # overlap between the patches in pixels (keep it at least large as gSiz)
  'only_init': True,    # set it to True to run CNMF-E
  'nb': 1, # number of background components (rank) if positive,
#             else exact ring model with following settings
#             gnb= 0: Return background as b and W
#             gnb=-1: Return full rank background B
#             gnb<-1: Don't return background

  'nb_patch': 0,# number of background components (rank) per patch if gnb>0
  'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
  'low_rank_background': None, # leaves background of each patch intact True performs global low-rank approximation if gnb>0

  'update_background_components': True,  # sometimes setting to False improve the results
  'min_corr': 0.8, # min peak value from correlation image
  'min_pnr': 8, # min peak to noise ration from PNR image
  'normalize_init': False,               # just leave as is
  'center_psf': True,                    # leave as is for 1 photon
  'ssub_B': 2, # downsampling factor for background
  'ring_size_factor': 1.4, # radius of ring is gSiz*ring_size_factor
  'del_duplicates': True,                # whether to remove duplicates from initialization
  'border_pix': 0 # number of pixels to not consider in the borders)
}
opts = params.CNMFParams(params_dict=opts_dict)


# In[11]:


analysis_start = time.time()
# Perform CNMF
cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=None, params=opts)
cnm.fit(images)

end = time.time()
print(end - analysis_start)


# ## Component Evaluation
# The components are evaluated in three ways:
# - the shape of each component must be correlated with the data
# - a minimum peak SNR is required over the length of a transient
# - each shape passes a CNN based classifier

# In[12]:


#%% COMPONENT EVALUATION


min_SNR = 3            # adaptive way to set threshold on the transient size
r_values_min = 0.8    # threshold on space consistency (if you lower more components
#                        will be accepted, potentially with worst quality)
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

print(' ***** ')
print('Number of total components: ', len(cnm.estimates.C))
print('Number of accepted components: ', len(cnm.estimates.idx_components))


# ## Plot results

# In[13]:


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

plt.figure(figsize=(30,15))
#plt.subplot(341);
#plt.subplot(345); plt.plot(mc.shifts_rig); plt.title('Motion corrected shifts')
plt.subplot(3,4,9);
plt.subplot(3,4,2); plt.imshow(cn_filter); plt.colorbar(); plt.title('Correlation projection')
plt.subplot(3,4,6); plt.imshow(pnr); plt.colorbar(); plt.title('PNR')
plt.subplot(3,4,10); plt.imshow(np.amax(SFP,axis=2)); plt.colorbar(); plt.title('Spatial footprints')

plt.subplot(2,2,2); plt.figure; plt.title('Example traces (first 50 cells)')
plot_gain = 10 # To change the value gain of traces
if numNeurons >= neuronsToPlot:
  for i in range(neuronsToPlot):
    if i == 0:
      plt.plot(RawTraces[i,:],'k')
    else:
      trace = RawTraces[i,:] + maxRawTraces*i/plot_gain
      plt.plot(trace,'k')
else:
  for i in range(numNeurons):
    if i == 0:
      plt.plot(RawTraces[i,:],'k')
    else:
      trace = RawTraces[i,:] + maxRawTraces*i/plot_gain
      plt.plot(trace,'k')

plt.subplot(2,2,4); plt.figure; plt.title('Deconvolved traces (first 50 cells)')
plot_gain = 20 # To change the value gain of traces
if numNeurons >= neuronsToPlot:
  for i in range(neuronsToPlot):
    if i == 0:
      plt.plot(DeconvTraces[i,:],'k')
    else:
      trace = DeconvTraces[i,:] + maxRawTraces*i/plot_gain
      plt.plot(trace,'k')
else:
  for i in range(numNeurons):
    if i == 0:
      plt.plot(DeconvTraces[i,:],'k')
    else:
      trace = DeconvTraces[i,:] + maxRawTraces*i/plot_gain
      plt.plot(trace,'k')

# Save summary figure
plt.savefig(result_data_dir + '/' + 'summary_figure.svg', edgecolor='w', format='svg', transparent=True)


# ## Register the timestamps for analysis

# In[ ]:


mstime = []
sessionLengths = []
for vid_dir in session_local_dir.keys():
  with open(vid_dir + '/timestamp.dat') as f:
    camNum, frameNum, sysClock, buffer = np.loadtxt(f, dtype='float', comments='#', skiprows=1, unpack = True)
  camNumber = camNum[0]
  mstime_idx = np.where(camNum == camNumber)
  mstime = mstime + sysClock[mstime_idx]
  sessionLengths.append(length(mstime_idx))
mstime[0] = 0


# In[ ]:


analysis_end = time.time()

analysis_duration = analysis_end - analysis_start

print('Done analyzing. This took a total ' + str(analysis_duration) + ' s')


# ## Save analysis

# In[ ]:


"""# Save the results in HDF5 format"""

save_hdf5 = True
if save_hdf5:
  cnm.save(result_data_dir + 'analysis_results.hdf5')

"""# Save the results in Matlab format"""

save_mat = True
if save_mat:
  from scipy.io import savemat

  results_dict = {
                #'dirName': path_to_analyze,
                'numFiles': 1,
                'framesNum': len(RawTraces[1]),
                'maxFramesPerFile': 1000,
                'height': dims[0],
                'width': dims[1],
                'camNumber': camNumber,
                 #'time': mstime,
                #'sessionLengths': sessionLengths,
                #'analysis_time': analysis_time,
                'meanFrame': [], #TO DO
                'Centroids': [], #TO DO
                'CorrProj': cn_filter,
                'PeakToNoiseProj': pnr,
                'RawTraces': RawTraces.conj().transpose(), #swap time x neurons dimensions
                #'FiltTraces': cnm.estimates.F_dff,
                #'DeconvTraces': cnm.estimates.S.conj().transpose(),
                'SFPs': SFP,
                'numNeurons': SFP_dims[2],
                #'analysis_duration': analysis_duration
                }

  SFPperm = np.transpose(SFP,[2,0,1])
  sio.savemat(result_data_dir + '/SFP.mat', {'SFP': SFPperm})
  sio.savemat(result_data_dir + '/ms.mat', {'ms': results_dict})


# In[ ]:


cnm.estimates.F_dff.shape


# In[ ]:


# Stop the cluster
cm.stop_server(dview=dview)

