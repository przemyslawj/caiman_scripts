import yaml

import miniscope_file
from load_args import *
import caiman as cm

info = dict()
info['experiment_title'] = experiment_title
info['experiment_date'] = experiment_date
info['animal_name'] = animal_name
info['spatial_downsampling'] = spatial_downsampling
info['downsample_subpath'] = downsample_subpath
info['local_rootdir'] = local_rootdir

session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, info['animal_name'])
info['session_fpaths'] = session_fpaths

timestamp_files = []
session_lengths = []
for s_fpath in session_fpaths:
    vids_fpath = miniscope_file.list_vidfiles(s_fpath, vid_prefix)
    total_frames = 0
    memmap_files = miniscope_file.get_memmap_files(s_fpath, doPwRigid, vid_prefix)
    for memmap_file in memmap_files:
        Yr, dim, T = cm.load_memmap(memmap_file, 'r')
        total_frames += T
    session_lengths.append(total_frames)
    timestamp_files.append(miniscope_file.get_timestamps_fpath(s_fpath))

info['session_lengths'] = session_lengths
info['timestamp_files'] = timestamp_files

with open(caiman_result_dir + '/session_info.yaml', 'w') as f:
    yaml.dump(info, f)
