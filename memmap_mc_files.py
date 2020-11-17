# Map motion corrected files to memory
#
import miniscope_file
from load_args import *

import caiman as cm
import os
import subprocess
import yaml

from caiman.cluster import setup_cluster

writeAvi = False
border_nan = 'copy'
session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, animal_name)

mc_fnames = []
max_bord_px = 0
for s_fpath in session_fpaths:
    session_memmap = miniscope_file.get_memmap_files(s_fpath, doPwRigid, vid_prefix)
    session_vids = miniscope_file.list_vidfiles(s_fpath, vid_prefix)
    mc_stats_fpath = miniscope_file.get_miniscope_vids_path(s_fpath) + '/mc_stats.yaml'
    if not os.path.isfile(mc_stats_fpath):
        raise FileNotFoundError('Missing file for motion correction stats for session: ' + s_fpath)
    if len(session_memmap) < len(session_vids):
        raise Exception('Some files not motion corrected for session: ' + s_fpath)
    mc_fnames = mc_fnames + session_memmap
    with open(mc_stats_fpath, 'r') as f:
        session_info = yaml.load_all(f, Loader=yaml.FullLoader)
    max_shift = 0 if border_nan is 'copy' else session_info['max_shift']
    max_bord_px = max(max_bord_px, max_shift)

c, dview, n_processes = setup_cluster(backend='local', n_processes=None, single_thread=False)
fname_new = cm.save_memmap(mc_fnames, base_name='memmap_', order='C',
                           border_to_0=max_bord_px)
print('Motion corrected videos has been mapped to single file')
cm.stop_server(dview=dview)

subprocess.call(['mkdir', '-p', caiman_result_dir])
output_file = os.path.join(caiman_result_dir, os.path.basename(fname_new))
subprocess.call(['mv', fname_new, output_file])

if writeAvi:
    import video
    video.write_avi(output_file, caiman_result_dir)
