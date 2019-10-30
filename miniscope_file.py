import logging
import os
import re
import subprocess
import yaml


def list_session_dirs(src_miniscope_path, animal_name):
    session_dirs = []
    exp_subdirs = ['trial', 'homecage', 'test']
    for exp_subdir in exp_subdirs:
        if not os.path.isdir(os.path.join(src_miniscope_path, exp_subdir)):
            continue
        sessions_rootdir = '/'.join([src_miniscope_path, exp_subdir, 'mv_caimg', animal_name])
        if os.path.isdir(sessions_rootdir):
            # create a list of vids to process
            sessions_list = [s for s in os.listdir(sessions_rootdir) if s.startswith('Session')]
            sessions_list = sorted(sessions_list, key=lambda x: int(re.sub('[Session]', '', x)))
            sessions_fpaths = [sessions_rootdir + '/' + s for s in sessions_list]
            session_dirs = session_dirs + sessions_fpaths

    return(session_dirs)


def get_timestamped_path(session_fpath):
    timestamped_dir = [f for f in os.listdir(session_fpath) if f.startswith('H')][0]
    timestamped_path = '/'.join([session_fpath, timestamped_dir])
    return(timestamped_path)


def sort_mscam(x: str):
    filename = os.path.basename(x)
    if not filename.startswith('msCam'):
        raise Exception('Expected msCam file, but got: ' + x)
    return int(re.findall("\d+", filename)[0])


def list_vidfiles(session_fpath):
    timestamped_path = get_timestamped_path(session_fpath)

    msFileList = [f for f in os.listdir(timestamped_path) if f.startswith('msCam') and f.endswith('.avi')]
    msFileList = sorted(msFileList, key=sort_mscam)
    vid_fpaths = [timestamped_path + '/' + fname for fname in msFileList]

    return vid_fpaths


def get_timestamp_dat_fpath(session_fpath):
    timestamped_path = get_timestamped_path(session_fpath)
    return timestamped_path + '/' + 'timestamp.dat'


def get_memmap_files(s_fpath, pwRigid=False, prefix='msCam'):
    infix = '_rig_'
    if pwRigid:
        infix = '_els_'
    timestamped_path = get_timestamped_path(s_fpath)
    mmapFiles = [timestamped_path + '/' + f for f in os.listdir(timestamped_path)
            if f.startswith(prefix) and f.endswith('.mmap') and infix in f]
    return sorted(mmapFiles, key=sort_mscam)


def get_joined_memmap_fpath(result_data_dir):
    fs = [result_data_dir + '/' + f for f in os.listdir(result_data_dir)
                    if f.startswith('memmap') and f.endswith('.mmap')]
    if len(fs) == 0:
        raise FileNotFoundError('No memmap file found at ' + result_data_dir)
    return fs[0]


def gdrive_download_file(gdrive_fpath, local_dir, rclone_config):
    logging.info('Downloading file: ' + gdrive_fpath + ' to: ' + local_dir)
    subprocess.run(['mkdir', '-p', local_dir])
    cp = subprocess.run(['rclone', 'copy', '-P', '--config', 'env/rclone.conf',
                        rclone_config + ':' + gdrive_fpath,
                        local_dir], capture_output=True, text=True)
    if cp.returncode != 0:
        logging.error('Failed to download: ' + gdrive_fpath + ' error: ' + str(cp.stderr))


def gdrive_upload_file(local_fpath, gdrive_dir, rclone_config):
    logging.info('Uploading file: ' + local_fpath + ' to: ' + gdrive_dir)
    cp = subprocess.run(['rclone', 'copy', '-P', '--config', 'env/rclone.conf',
                         local_fpath,
                         rclone_config + ':' + gdrive_dir],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        logging.error('Failed to upload to: ' + gdrive_dir + ' error: ' + str(cp.stderr))


def load_session_info(result_dir, gdrive_result_dir, rclone_config):
    session_info_fpath = os.path.join(result_dir, 'session_info.yaml')
    if not os.path.isfile(session_info_fpath):
        gdrive_download_file(gdrive_result_dir + '/session_info.yaml', result_dir, rclone_config)
    with open(session_info_fpath, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_hdf5_result(result_dir, gdrive_result_dir, rclone_config):
    h5fpath = os.path.join(result_dir, 'analysis_results.hdf5')
    if not os.path.isfile(h5fpath):
        gdrive_download_file(gdrive_result_dir + '/analysis_results.hdf5', result_dir, rclone_config)

    from caiman.source_extraction.cnmf.cnmf import load_CNMF
    return load_CNMF(h5fpath)


def load_msresult(result_dir, gdrive_result_dir, rclone_config):
    ms_path = os.path.join(result_dir, 'ms.mat')
    if not os.path.isfile(ms_path):
        gdrive_download_file(gdrive_result_dir + '/ms.mat', result_dir, rclone_config)
    from scipy.io import loadmat
    return loadmat(ms_path)

