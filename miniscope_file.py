import os
import re

def list_session_dirs(src_miniscope_path, animal_name):
    session_dirs = []
    for exp_subdir in os.listdir(src_miniscope_path):
        if exp_subdir.startswith('caiman'):
            continue
        sessions_rootdir = '/'.join([src_miniscope_path, exp_subdir, 'mv_caimg', animal_name])

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


def get_memmap_files(s_fpath, rigid=True, prefix='msCam'):
    infix = '_els_'
    if rigid:
        infix = '_rig_'
    timestamped_path = get_timestamped_path(s_fpath)
    mmapFiles = [timestamped_path + '/' + f for f in os.listdir(timestamped_path)
            if f.startswith(prefix) and f.endswith('.mmap') and infix in f]
    return sorted(mmapFiles, key=sort_mscam)


def get_joined_memmap_fpath(result_data_dir):
    return [result_data_dir + '/' + f for f in os.listdir(result_data_dir)
                    if f.startswith('memmap') and f.endswith('.mmap')][0]
