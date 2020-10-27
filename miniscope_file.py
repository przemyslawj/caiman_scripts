import logging
import os
import re
import subprocess
import yaml

from load_args import *


def list_session_dirs(src_miniscope_path, animal_name):
    session_dirs = []
    exp_subdirs = ['trial', 'homecage','beforetest', 'aftertest']
    logging.info('Listing sessions in dir=%s for animal=%s', src_miniscope_path, animal_name)
    for exp_subdir in exp_subdirs:
        exp_path = os.path.join(src_miniscope_path, exp_subdir)
        if not os.path.isdir(exp_path):
            continue
        if miniscope_v4:
            sessions_rootdir = os.path.join(exp_path, animal_name)
        else:
            sessions_rootdir = os.path.join(exp_path, 'mv_caimg', animal_name)
        if os.path.isdir(sessions_rootdir):
            # create a list of vids to process
            sessions_list = [s for s in os.listdir(sessions_rootdir) if s.startswith('Session')]
            sessions_list = sorted(sessions_list, key=lambda x: int(re.sub('[Session]', '', x)))
            sessions_fpaths = [sessions_rootdir + '/' + s for s in sessions_list]
            session_dirs = session_dirs + sessions_fpaths

    return(session_dirs)


def _get_timestamped_path(session_fpath):
    timestamped_dir = [f for f in os.listdir(session_fpath) if f[0] == 'H' or f[0].isdigit()][0]
    timestamped_path = '/'.join([session_fpath, timestamped_dir])
    return(timestamped_path)


def sort_mscam(vid_prefix: str):
    def sort_fun(x: str):
        filename = os.path.basename(x)
        if not filename.startswith(vid_prefix):
            raise Exception('Expected avi files, but got: ' + x)
        return int(re.findall("\d+", filename)[0])
    return sort_fun

def get_miniscope_vids_path(session_fpath: str):
    timestamped_path = _get_timestamped_path(session_fpath)
    miniscope_vids_path = timestamped_path
    if miniscope_v4:
        miniscope_vids_path = os.path.join(timestamped_path, 'Miniscope')
    return miniscope_vids_path


def list_vidfiles(session_fpath: str, vid_prefix: str):
    miniscope_vids_path = get_miniscope_vids_path(session_fpath)
    msFileList = [f for f in os.listdir(miniscope_vids_path) if f.startswith(vid_prefix) and f.endswith('.avi')]
    msFileList = sorted(msFileList, key=sort_mscam(vid_prefix))
    return [os.path.join(miniscope_vids_path, fname) for fname in msFileList]


def get_timestamps_fpath(session_fpath):
    timestamped_path = get_miniscope_vids_path(session_fpath)
    v4_path = timestamped_path + '/' + 'timeStamps.csv'
    if os.path.exists(v4_path):
        return v4_path
    return timestamped_path + '/' + 'timestamp.dat'


def get_memmap_files(s_fpath, pwRigid: bool, vid_prefix: str):
    infix = '_rig_'
    if pwRigid:
        infix = '_els_'
    timestamped_path = get_miniscope_vids_path(s_fpath)
    mmapFiles = [timestamped_path + '/' + f for f in os.listdir(timestamped_path)
            if f.startswith(vid_prefix) and f.endswith('.mmap') and infix in f]
    return sorted(mmapFiles, key=sort_mscam(vid_prefix))


def get_joined_memmap_fpath(result_data_dir):
    fs = [result_data_dir + '/' + f for f in os.listdir(result_data_dir)
                    if f.startswith('memmap') and f.endswith('.mmap')]
    if len(fs) == 0:
        raise FileNotFoundError('No memmap file found at ' + result_data_dir)
    return fs[0]

def mkdir(dirpath):
    subprocess.run(['mkdir', '-p', dirpath])

def gdrive_download_file(gdrive_fpath, local_dir, rclone_config):
    logging.info('Downloading file: ' + gdrive_fpath + ' to: ' + local_dir)
    mkdir(local_dir)
    src_fpath = rclone_config + '' + gdrive_fpath
    cp = subprocess.run(['rclone', 'copy', '-P', '--config', 'env/rclone.conf',
                        src_fpath,
                        local_dir], capture_output=True, text=True)
    if cp.returncode != 0:
        logging.error('Failed to download: ' + gdrive_fpath + ' error: ' + str(cp.stderr))
        return False
    return True


def gdrive_upload_file(local_fpath, gdrive_dir, rclone_config):
    logging.info('Uploading file: ' + local_fpath + ' to: ' + gdrive_dir)
    target_dir = rclone_config  + ':' + gdrive_dir
    cp = subprocess.run(['rclone', 'copy', '-P', '--config', 'env/rclone.conf',
                         local_fpath,
                         target_dir],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        logging.error('Failed to upload to: ' + gdrive_dir + ' error: ' + str(cp.stderr))
        return False
    return True


def load_session_info(result_dir, gdrive_result_dir, rclone_config):
    session_info_fpath = os.path.join(result_dir, 'session_info.yaml')
    if not os.path.isfile(session_info_fpath):
        gdrive_download_file(gdrive_result_dir + '/session_info.yaml', result_dir, rclone_config)
    with open(session_info_fpath, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_hdf5_result(result_dir, gdrive_result_dir, rclone_config, use_filtered=False):
    hdf5_filename = 'analysis_results.hdf5'
    if use_filtered:
        hdf5_filename = 'analysis_results_filtered.hdf5'
        result_dir = os.path.join(result_dir, 'filtered')
        gdrive_result_dir = os.path.join(gdrive_result_dir, 'filtered')
    h5fpath = os.path.join(result_dir, hdf5_filename)
    if not os.path.isfile(h5fpath):
        success = gdrive_download_file(os.path.join(gdrive_result_dir, hdf5_filename), result_dir, rclone_config)
        if not success:
            return None

    from caiman.source_extraction.cnmf.cnmf import load_CNMF
    try:
        return load_CNMF(h5fpath)
    except OSError as e:
        logging.error('Failed to open file: %s', h5fpath)
        raise e


def load_msresult(result_dir, gdrive_result_dir, rclone_config):
    ms_path = os.path.join(result_dir, 'ms.mat')
    if not os.path.isfile(ms_path):
        gdrive_download_file(gdrive_result_dir + '/ms.mat', result_dir, rclone_config)
    from scipy.io import loadmat
    return loadmat(ms_path)

