"""# Install and import dependencies"""
import miniscope_file

from datetime import datetime
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
import numpy as np
import subprocess

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params

writeAvi = True
doPwRigid = False

"""# Prepare data"""
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
result_data_dir = '/'.join([local_miniscope_path, 'caiman', animal_name])

now = datetime.now()
analysis_time = now.strftime("%Y-%m-%d %H:%M") # This is to register when the analysis was performed
print('Analysis started on ' + analysis_time)

analysis_start = time.time() # This is to register the time spent analyzing

#%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
#if 'dview' in locals():
#    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

"""# Set parameters for motion correction
Ideally, optimize these for your datasets then stick to these values
"""

# dataset dependent parameters
frate = 20                       # movie frame rate
decay_time = 0.4                 # length of a typical transient in seconds

# motion correction parameters
pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
gSig_filt = (3, 3)       # size of high pass spatial filtering, used in 1p data
max_shifts = (5, 5)      # maximum allowed rigid shift
strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
overlaps = (16, 16)      # overlap between patches (size of patch strides+overlaps)
max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
border_nan = 'copy'      # replicate values along the boundaries
use_cuda = True         # Set to True in order to use GPU
only_init_patch = True

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
    'memory_fact': 0.7
}

opts = params.CNMFParams(params_dict=mc_dict)

"""# Perform motion correction (might take a while)"""
def plot_stats(session_fpath, mc):
    # Plot the motion corrected template and associated shifts
    subprocess.call(['mkdir', '-p', result_data_dir])
    plt.figure(figsize=(10, 20))
    m_rig = cm.load(mc.mmap_file)
    plt.subplot(2, 1, 1)
    plt.imshow(m_rig.local_correlations(eight_neighbours=True, swap_dim=False))
    plt.subplot(2, 1, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
    plt.legend(['x shifts', 'y shifts'])
    plt.xlabel('frames')
    plt.ylabel('pixels')
    name_parts = session_fpath.split(os.path.sep)
    plt_fname = '_'.join([name_parts[-4], name_parts[-2], name_parts[-1], 'mc_summary.svg'])
    plt.savefig(result_data_dir + '/' + plt_fname, edgecolor='w', format='svg', transparent=True)


def mc_vids(vids_fpath, mc_rigid_template):
    # do motion correction rigid
    mc = MotionCorrect(vids_fpath, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True, template=mc_rigid_template)

    if doPwRigid:
        mc.pw_rigid = True
        mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)
        mc.motion_correct(save_movie=True, template=mc.total_template_rig)

    end = time.time()
    print('Motion correction done in ' + str(end - start))
    return(mc)


start = time.time() # This is to keep track of how long the analysis is running
max_bord_px = 0
mc_rigid_template = None
rigid_template_fpath = result_data_dir + '/mc_rigid_template'
if os.path.isfile(rigid_template_fpath + '.npy'):
    mc_rigid_template = np.load(rigid_template_fpath + '.npy', mc_rigid_template)

for s_fpath in session_fpaths:
    vids_fpath = miniscope_file.list_vidfiles(s_fpath)

    # If directory already processed
    memmap_files = miniscope_file.get_memmap_files(s_fpath)
    if len(memmap_files) >= len(vids_fpath) and (mc_rigid_template is not None):
        continue

    print('Aligning session vids:' + str(vids_fpath))
    mc = mc_vids(vids_fpath, mc_rigid_template)
    fname_mc = mc.fname_tot_els if doPwRigid else mc.fname_tot_rig
    print('Created motion corrected files: ' + str(fname_mc))

    if mc_rigid_template is None:
        mc_rigid_template = mc.total_template_rig
        np.save(rigid_template_fpath, mc_rigid_template)
    plot_stats(s_fpath, mc)

    """# Map the motion corrected video to memory"""

    if doPwRigid:
        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                        np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
    else:
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)

    bord_px = 0 if border_nan is 'copy' else bord_px
    max_bord_px = max(max_bord_px, bord_px)


mc_fnames = []
for s_fpath in session_fpaths:
    mc_fnames = mc_fnames + miniscope_file.get_memmap_files(s_fpath)

fname_new = cm.save_memmap(mc_fnames, base_name='memmap_', order='C',
                            border_to_0=max_bord_px)
print('Motion corrected videos has been mapped to single memory file')

subprocess.call(['mkdir', '-p', '/'.join([local_miniscope_path, 'caiman', animal_name])])
output_file = '/'.join([local_miniscope_path, 'caiman', os.path.basename(fname_new)])
subprocess.call(['mv', fname_new, output_file])

if writeAvi:
    import video
    video.write_avi(output_file, result_data_dir)

cm.stop_server(dview=dview)
