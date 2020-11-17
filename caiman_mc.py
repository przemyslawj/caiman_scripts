"""# Install and import dependencies"""
import miniscope_file
from load_args import *
import video

from datetime import datetime
import logging
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
import numpy as np
import yaml
import subprocess

import caiman as cm
from caiman.motion_correction import MotionCorrect


logging.basicConfig(level=logging.INFO)
shortRun = False
rerun = False

session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, experiment_date, animal_name)
if shortRun:
    session_fpaths = [session_fpaths[0]]
subprocess.call(['mkdir', '-p', caiman_result_dir])

now = datetime.now()
analysis_time = now.strftime("%Y-%m-%d %H:%M")
logging.info('Analysis started on %s', analysis_time)

analysis_start = time.time()

#if a cluster already exists it will be closed and a new session will be opened
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=ncores, single_thread=False)

"""# Set parameters for motion correction
Ideally, optimize these for your datasets then stick to these values
"""

# dataset dependent parameters
frate = 20                       # movie frame rate

# motion correction parameters
pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
#gSig_filt = (8, 8)       # size of high pass spatial filtering, used in 1p data
gSig_filt = (3, 3)       # size of high pass spatial filtering, used in 1p data
max_shifts = (6, 6)      # maximum allowed rigid shift
strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
overlaps = (16, 16)      # overlap between patches (size of patch strides+overlaps)
border_nan = 'copy'      # replicate values along the boundaries
use_cuda = False         # Set to True in order to use GPU
only_init_patch = True
splits_rig = 2
splits_els = 2
upsample_factor_grid = 4  # upsample factor to avoid smearing when merging patches
max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts


# ## Read CNMFE params
local_params_fpath = os.path.join(local_miniscope_path, 'cnmfe_params.csv')

if os.path.isfile(local_params_fpath):
    params_csv = pd.read_csv(local_params_fpath)
    animal_params = params_csv[params_csv['animal'] == animal_name]
else:
    animal_params = []
if len(animal_params) > 0:
    logging.info('Using CNMFE params file for motion correction')
    gSig_filt = (animal_params['gSig_filt'].values[0], animal_params['gSig_filt'].values[0])
else:
    logging.info('Using default CNMFE params')


def get_mean_frame(f):
    if f.endswith('.mmap'):
        m = video.load_images(f)
        return np.mean(m, 0)
    elif f.endswith('.avi'):
        return video.mean_frame_avi(f)
    else:
        raise ValueError('Extension not supported for file: ' + os.path.basename(f))


def calc_crispness(mean_frame):
    return float(np.sqrt(
        np.sum(np.sum(np.array(np.gradient(mean_frame)) ** 2, 0))))


def eval_mc_quality(fnames):
    if len(fnames) == 0:
        return [], []

    joint_crispness_list = []
    crispness_list = []
    mf1 = get_mean_frame(fnames[0])
    for f1, f2 in zip(fnames[:-1], fnames[1:]):
        if mf1 is None:
            mf1 = get_mean_frame(f1)
        crispness_list.append(calc_crispness(mf1))
        mf2 = get_mean_frame(f2)
        joint_crispness_list.append(calc_crispness((mf1 + mf2) / 2))
        mf1 = mf2
    crispness_list.append(calc_crispness(mf1))
    return crispness_list, joint_crispness_list


"""# Perform motion correction (might take a while)"""
def plot_stats(session_fpath, mc, shifts_rig):
    # Plot the motion corrected template and associated shifts
    plt.figure(figsize=(10, 20))
    nplots = 3 if doPwRigid else 2
    plt.subplot(nplots, 1, 1)
    # m_rig = cm.load(mc.mmap_file)
    # plt.imshow(m_rig.local_correlations(eight_neighbours=True, swap_dim=False))
    plt.imshow(mc.total_template_rig)

    plt.subplot(nplots, 1, 2)
    plt.plot(shifts_rig)  # % plot rigid shifts
    plt.legend(['rig x shifts', 'rig y shifts'])
    plt.xlabel('frames')
    plt.ylabel('pixels')
    if doPwRigid:
        plt.subplot(nplots, 1, 3)
        plt.plot(np.max(mc.x_shifts_els, axis=1))
        plt.plot(np.max(mc.y_shifts_els, axis=1))
        plt.legend(['max pw x shifts', 'max pw y shifts'])
        plt.xlabel('frames')
        plt.ylabel('pixels')

    name_parts = session_fpath.split(os.path.sep)
    plt_fname = '_'.join([name_parts[-4], name_parts[-3], name_parts[-2], name_parts[-1], 'mc_summary.svg'])
    plt.savefig(caiman_result_dir + '/' + plt_fname, edgecolor='w', format='svg', transparent=True)


def mc_vids(vids_fpath, mc_rigid_template):
    start = time.time()
    # estimated minimum value of the movie to produce an output that is positive
    min_mov = np.array([cm.motion_correction.high_pass_filter_space(
        m_, gSig_filt) for m_ in cm.load(vids_fpath[0], subindices=range(400))]).min()
    mc = MotionCorrect(
        vids_fpath,
        min_mov,
        dview=dview,
        max_shifts=max_shifts,
        niter_rig=1,
        splits_rig=splits_rig,
        num_splits_to_process_rig=None,
        shifts_opencv=True,
        nonneg_movie=True,
        gSig_filt=gSig_filt,
        border_nan=border_nan,
        is3D=False)

    mc.motion_correct_rigid(save_movie=(not doPwRigid), template=mc_rigid_template)

    shifts_rig = mc.shifts_rig
    template_rig = mc.total_template_rig

    if doPwRigid:
        mc.motion_correct_pwrigid(save_movie=True, template=template_rig)
        mc.total_template_rig = template_rig

    duration = time.time() - start
    logging.info('Motion correction done in %s', str(duration))
    return mc, duration, shifts_rig


max_bord_px = 0
mc_rigid_template = None
rigid_template_fpath = caiman_result_dir + '/mc_rigid_template'
if os.path.isfile(rigid_template_fpath + '.npy') and not rerun:
    mc_rigid_template = np.load(rigid_template_fpath + '.npy')

logging.info('Sessions to motion correct %d', len(session_fpaths))
for s_fpath in session_fpaths:
    session_vids = miniscope_file.list_vidfiles(s_fpath, vid_prefix)
    if shortRun:
        session_vids = [session_vids[0]]
    logging.info('Correcting %d vids in session %s',
            len(session_vids), s_fpath)
    mc_stats_fpath = miniscope_file.get_miniscope_vids_path(s_fpath) + '/mc_stats.yaml'

    # If directory already processed
    memmap_files = miniscope_file.get_memmap_files(s_fpath, doPwRigid, vid_prefix)
    if (len(memmap_files) >= len(session_vids)
            and (mc_rigid_template is not None)
            and (os.path.isfile(mc_stats_fpath)))\
            and (not rerun):
        continue

    logging.debug('Aligning session vids: %s', str(session_vids))
    mc, duration, shifts_rig = mc_vids(session_vids, mc_rigid_template)
    fname_mc = mc.fname_tot_els if doPwRigid else mc.fname_tot_rig
    logging.debug('Created motion corrected files: %s', str(fname_mc))

    if mc_rigid_template is None:
        mc_rigid_template = mc.total_template_rig
        np.save(rigid_template_fpath, mc_rigid_template)
    plot_stats(s_fpath, mc, shifts_rig)

    if doPwRigid:
        max_shift = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                       np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
    else:
        max_shift = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)

    end_time = time.time()
    mc_stats = dict()
    mc_stats['analysed_datetime'] = analysis_time
    mc_stats['mc_duration'] = duration
    mc_stats['max_shift'] = int(max_shift)
    mc_stats['PwRigid'] = doPwRigid
    vids_crispness, joint_vids_crispness = eval_mc_quality(session_vids)
    mc_stats['crispness_before'] = vids_crispness
    mc_stats['crispness_before_movie_pairs'] = joint_vids_crispness
    vids_crispness, joint_vids_crispness = eval_mc_quality(fname_mc)
    mc_stats['crispness_after'] = vids_crispness
    mc_stats['crispness_after_movie_pairs'] = joint_vids_crispness
    with open(mc_stats_fpath, 'w') as f:
        yaml.dump(mc_stats, f)

    # Restart server to clear memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=n_processes, single_thread=False)

cm.stop_server(dview=dview)

