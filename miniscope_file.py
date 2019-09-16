import os
import re


def list_session_dirs(src_miniscope_path, animal_name):
    session_dirs = []
    for exp_subdir in os.listdir(src_miniscope_path):
        if exp_subdir == 'caiman':
            continue
        sessions_rootdir = '/'.join([src_miniscope_path, exp_subdir, 'mv_caimg', animal_name])

        # create a list of vids to process
        sessions_list = [s for s in os.listdir(sessions_rootdir) if s.startswith('Session')]
        sessions_list = sorted(sessions_list, key=lambda x: int(re.sub('[Session]', '', x)))
        sessions_fpaths = [sessions_rootdir + '/' + s for s in sessions_list]
        session_dirs = session_dirs + sessions_fpaths

    return(session_dirs)


def list_vidfiles(session_fpath):
    timestamped_dir = os.listdir(session_fpath)[0]
    timestamped_path = '/'.join([session_fpath, timestamped_dir])

    msFileList = [f for f in os.listdir(timestamped_path) if f.startswith('ms')]
    msFileList = sorted(msFileList, key=lambda x: int(re.sub('[msCam.avi]', '', x)))
    vid_fpaths = [timestamped_path + '/' + fname for fname in msFileList]

    return vid_fpaths
