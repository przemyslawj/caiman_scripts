experiment_month = '2019-08'
experiment_title = 'habituation'
experiment_date = '2019-08-27' #@param {type: "string"}
animal_name = 'E-BL'  #@param {type: "string"}

"""# Install and import dependencies"""

from datetime import datetime
import scipy.io as sio
import re
import os
import h5py
import csv
import tensorflow as tf
import time
import logging
import zipfile
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
import numpy as np
from moviepy.editor import *
import smtplib


import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
import peakutils

"""# Prepare data"""

import os
import subprocess

copy_src_files = True
downsample = True
motion_correct = True


local_miniscope_path = '/'.join(['/mnt/DATA/Prez/cheeseboard-results/', experiment_month, experiment_title, experiment_date])
src_miniscope_path = '/'.join(['/mnt/DATA/Prez/cheeseboard/', experiment_month, experiment_title, experiment_date])
session_vids = dict()
session_local_dir = dict()
vid_fpaths = []
for exp_subdir in os.listdir(src_miniscope_path):
  if exp_subdir == 'caiman':
    continue
  local_session_path = '/'.join([local_miniscope_path, exp_subdir, animal_name])
  sessions_path = '/'.join([src_miniscope_path, exp_subdir, 'mv_caimg', animal_name])

  target_dir = local_miniscope_path + '/' + exp_subdir
  if copy_src_files or (not os.path.exists(target_dir)):
    print('Copying session ' + sessions_path + ' to ' + target_dir)
    subprocess.run(['mkdir', '-p', local_session_path])
    subprocess.run(['cp', '-r', sessions_path, target_dir])

  # create a list of vids to process
  sessions_list = [s for s in os.listdir(sessions_path) if s.startswith('Session')]
  sessions_list = sorted(sessions_list, key=lambda x: int(re.sub('[Session]','', x)))
  for session_subdir in sessions_list:
    timestamped_dir = os.listdir(sessions_path + '/' + session_subdir)[0]
    timestamped_path = '/'.join([sessions_path, session_subdir, timestamped_dir])

    msFileList = [f for f in os.listdir(timestamped_path) if f.startswith('ms')]
    msFileList = sorted(msFileList, key=lambda x: int(re.sub('[msCam.avi]','', x)))
    exp_id = '_'.join([exp_subdir, animal_name, session_subdir, timestamped_dir])
    local_vids_parentdir = '/'.join([local_session_path, session_subdir, timestamped_dir])
    session_vids[exp_id] = [local_vids_parentdir + '/' + fname for fname in msFileList]
    session_local_dir[exp_id] = local_vids_parentdir
    vid_fpaths = vid_fpaths + session_vids[exp_id]

result_data_dir = local_miniscope_path + '/caiman/' + animal_name + '/'
subprocess.run(['mkdir', '-p', result_data_dir])
#vid_fpaths=vid_fpaths[1:2]


print(session_vids)

"""# Setup cnmf parameters"""

analyze_behavior = False
spatial_downsampling = 2 # Drastically speeds up processing. 2-3 recommended
isnonrigid = False

now = datetime.now()
analysis_time = now.strftime("%Y-%m-%d %H:%M") # This is to register when the analysis was performed
print('Analysis started on ' + analysis_time)

analysis_start = time.time() # This is to register the time spent analyzing


"""# Downsample the videos"""
if downsample:
  for video in vid_fpaths:
    clip = VideoFileClip(video)
    resized_clip = clip.resize(1/spatial_downsampling)
    os.remove(video)
    resized_clip.write_videofile(video,codec='rawvideo')

#%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

"""# Set parameters for motion correction
Ideally, optimize these for your datasets then stick to these values
"""

# dataset dependent parameters
frate = 20                       # movie frame rate
decay_time = 0.4                 # length of a typical transient in seconds

# motion correction parameters
pw_rigid = True         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
gSig_filt = (3, 3)       # size of high pass spatial filtering, used in 1p data
max_shifts = (5, 5)      # maximum allowed rigid shift
strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
overlaps = (16, 16)      # overlap between patches (size of patch strides+overlaps)
max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
border_nan = 'copy'      # replicate values along the boundaries
use_cuda = True         # Set to True in order to use GPU
only_init_patch = True
memory_fact = 1.0

mc_dict = {
    #'fnames': vid_fpaths,
    'fr': frate,
    'niter_rig': 1,
    'splits_rig': 20,  # for parallelization split the movies in  num_splits chuncks across time
    # if none all the splits are processed and the movie is saved
    'num_splits_to_process_rig': None, # intervals at which patches are laid out for motion correction
    'decay_time': decay_time,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'gSig_filt': gSig_filt,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': border_nan,
    'use_cuda' : use_cuda,
    'only_init_patch' : only_init_patch,
    'memory_fact': memory_fact
}

opts = params.CNMFParams(params_dict=mc_dict)

"""# Perform motion correction (might take a while)"""

start = time.time() # This is to keep track of how long the analysis is running
if motion_correct:
    # do motion correction rigid
    mc = MotionCorrect(vid_fpaths, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig

end = time.time()

print(end - start)
print('Motion correction has been done!')

"""# Plot the motion corrected template and associated shifts"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
if motion_correct and not pw_rigid:
  plt.figure(figsize=(10,20))
  plt.subplot(2, 1, 1); plt.imshow(mc.total_template_rig);  # % plot template
  plt.subplot(2, 1, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
  plt.legend(['x shifts', 'y shifts'])
  plt.xlabel('frames')
  plt.ylabel('pixels')
  plt.savefig(result_data_dir + '/' + 'mc_summary_figure.svg', edgecolor='w', format='svg', transparent=True)

"""# Map the motion corrected video to memory"""

if motion_correct:
    if pw_rigid:
        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
    else:
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)

    bord_px = 0 if border_nan is 'copy' else bord_px
    fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                               border_to_0=bord_px)

else:  # if no motion correction just memory map the file
    fname_new = cm.save_memmap(vid_fpaths, base_name='memmap_',
                               order='C', border_to_0=0, dview=dview)

print('Motion corrected video has been mapped to memory')

# load memory mappable file
print('New mappable file: ' + fname_new)
Yr, dims, T = cm.load_memmap(fname_new)
images = Yr.T.reshape((T,) + dims, order='F')

# Write motion corrected video to drive
w = cm.movie(images)
w.save(result_data_dir + '/mc.tif')

# using skvideo
#import skvideo.io
#mcwriter = skvideo.io.FFmpegWriter(result_data_dir + '/mc.mp4', outputdict={
#  '-c:v': 'copy'})
#mcwriter = skvideo.io.FFmpegWriter(result_data_dir + '/mc.avi')
#
#for iddxx, frame in enumerate(w):
#  mcwriter.writeFrame(frame.astype('uint8'))

#mcwriter.close()
cm.stop_server(dview=dview)

