import yaml

import miniscope_file
from load_args import *
import caiman as cm

info = dict()
info['experiment_month'] = experiment_month
info['experiment_title'] = experiment_title
info['experiment_date'] = experiment_date
info['animal_name'] = animal_name
info['spatial_downsampling'] = spatial_downsampling
info['downsample_subpath'] = downsample_subpath
info['local_rootdir'] = local_rootdir

session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, info['animal_name'])
info['session_fpaths'] = session_fpaths

dat_files = []
session_lengths = []
for s_fpath in session_fpaths:
    vids_fpath = miniscope_file.list_vidfiles(s_fpath)
    total_frames = 0
    memmap_files = miniscope_file.get_memmap_files(s_fpath)
    for memmap_file in memmap_files:
        Yr, dim, T = cm.load_memmap(memmap_file, 'r')
        total_frames += T
    session_lengths.append(total_frames)
    dat_files.append(miniscope_file.get_timestamp_dat_fpath(s_fpath))

info['session_lengths'] = session_lengths
info['dat_files'] = dat_files

with open(result_data_dir + '/session_info.yaml', 'w') as f:
    yaml.dump(info, f)