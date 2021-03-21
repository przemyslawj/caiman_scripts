import logging
import re
import subprocess
import yaml

from load_args import *


def _recursively_visit_dirs(parent_dir, depth):
    if not os.path.isdir(parent_dir):
        return []
    if depth == 0:
        return [parent_dir]
    res = []
    for subdir in sorted(os.listdir(parent_dir)):
        res += _recursively_visit_dirs(os.path.join(parent_dir, subdir), depth-1)
    return res


def list_all_session_dirs(local_rootdir):
    depth = len(pipeline_setup['directoryStructure'])
    return _recursively_visit_dirs(local_rootdir, depth)


def _path_parts(path):
    p, f = os.path.split(path)
    return _path_parts(p) + [f] if f else [p]


def get_session_dirs_df(local_rootdir):
    import pandas as pd
    nsub_dirs = len(pipeline_setup['directoryStructure'])
    df = pd.DataFrame({'path': list_all_session_dirs(local_rootdir)})
    for i, var in enumerate(pipeline_setup['directoryStructure']):
        df[var] = [_path_parts(p)[-(nsub_dirs - i)] for p in df['path'] if len(_path_parts(p)) > nsub_dirs]
    return df


def list_session_dirs(local_rootdir, experiment_date, animal_name):
    df = get_session_dirs_df(local_rootdir)
    paths = df[df.date.eq(experiment_date) & df.animalID.eq(animal_name)].path
    return list(paths.values)


def sort_mscam(vid_prefix: str):
    def sort_fun(x: str):
        filename = os.path.basename(x)
        if not filename.startswith(vid_prefix):
            raise Exception('Expected avi files, but got: ' + x)
        return int(re.findall("\d+", filename)[0])
    return sort_fun


def get_miniscope_vids_path(session_fpath: str):
    if is_v4_path:
        return os.path.join(session_fpath, 'Miniscope')
    return session_fpath


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


# Prints the process stdout until it is terminated
def _print_process_stdout(cp):
    while True:
        output = cp.stdout.readline()
        if output == '' and cp.poll() is not None:
            break
        if output:
            print(output.strip())


def gdrive_download_file(gdrive_fpath, local_dir, rclone_config):
    logging.info('Downloading path: ' + gdrive_fpath + ' to: ' + local_dir)
    src_fpath = rclone_config + ':' + gdrive_fpath
    cp = subprocess.Popen(['rclone', 'copy', '-P', '--config', 'env/rclone.conf', src_fpath, local_dir],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          encoding='utf8')
    _print_process_stdout(cp)
    if cp.returncode == 0:
        return True

    logging.error('Failed to download: %s, error=%s', gdrive_fpath, cp.stderr.read())
    return False


def gdrive_upload_file(local_fpath, gdrive_dir, rclone_config):
    logging.info('Uploading path: ' + local_fpath + ' to: ' + gdrive_dir)
    target_dir = rclone_config + ':' + gdrive_dir
    cp = subprocess.Popen(['rclone', 'copy', '-P', '--config', 'env/rclone.conf', local_fpath, target_dir],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          encoding='utf8')
    _print_process_stdout(cp)
    if cp.returncode == 0:
        return True

    logging.error('Failed to upload dir=%s, error=%s', gdrive_dir, cp.stderr.read())
    return False


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

