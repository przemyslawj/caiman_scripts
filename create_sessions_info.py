import os
import yaml

import miniscope_file
import caiman as cm

info = dict()
info['experiment_month'] = os.environ['EXP_MONTH']
info['experiment_title'] = os.environ['EXP_TITLE']
info['experiment_date'] = os.environ['EXP_DATE']
info['animal_name'] = os.environ['ANIMAL']
info['spatial_downsampling'] = int(os.environ['DOWNSAMPLE'])
info['downsample_subpath'] = os.environ['DOWNSAMPLE_SUBPATH']
info['local_rootdir'] = os.environ['LOCAL_ROOTDIR']

local_miniscope_path = '/'.join([
    info['local_rootdir'],
    info['downsample_subpath'],
    info['experiment_month'],
    info['experiment_title'],
    info['experiment_date']])

result_data_dir = '/'.join([local_miniscope_path, 'caiman', info['animal_name']])

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