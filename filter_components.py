from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf import params as params
import caiman as cm

import cv2
import numpy as np
import os

import video
import logging

logging.basicConfig(level=logging.INFO)

from miniscope_file import gdrive_download_file, load_session_info

# Choose video
exp_month = '2019-08'
exp_title = 'learning'
exp_date = '2019-08-30'
animal = 'F-TL'
#rootdir = '/home/przemek/neurodata/'
rootdir = '/mnt/DATA/Prez/caiman_instance/Prez/'
gdrive_subdir = 'cheeseboard-down/down_2'
rclone_config = os.environ['RCLONE_CONFIG']

eval_params = {
    'cnn_lowest': .1,
    'min_cnn_thr': 0.9,
    'use_cnn': True,
    'rval_thr': 0.8,
    'rval_lowest': -1.0,
    'min_SNR': 6,
    'SNR_lowest': 2.5
}


""" Prepare data """
gdrive_dated_dir = os.path.join(gdrive_subdir, exp_month, exp_title, exp_date)
local_dated_dir = os.path.join(rootdir, gdrive_dated_dir)
gdrive_result_dir = os.path.join(gdrive_dated_dir, 'caiman', animal)
result_dir = os.path.join(local_dated_dir, 'caiman', animal)
# Download result files if not stored locally
h5fpath = os.path.join(result_dir, 'analysis_results.hdf5')
if not os.path.isfile(h5fpath):
    gdrive_download_file(gdrive_result_dir + '/analysis_results.hdf5', result_dir, rclone_config)
cnm_obj = load_CNMF(h5fpath)

session_info = load_session_info(result_dir, gdrive_result_dir, rclone_config)
opts = params.CNMFParams(params_dict=eval_params)
A = cnm_obj.estimates.A

cnm_obj.estimates.threshold_spatial_components(maxthr=0.5)
cnm_obj.estimates.remove_small_large_neurons(min_size_neuro=20, max_size_neuro=110)
idx_components_bad = cnm_obj.estimates.idx_components_bad
eval_params['use_cnn'] = False
cnm_obj.estimates.filter_components(images, cnm_obj.params, new_dict=eval_params)

cnm_obj.estimates.idx_components_bad = sorted(np.union1d(cnm_obj.estimates.idx_components_bad, idx_components_bad))
cnm_obj.estimates.idx_components = sorted(np.setdiff1d(np.arange(cnm_obj.estimates.A.shape[-1]), cnm_obj.estimates.idx_components_bad))
print('Bad components: ' + str(cnm_obj.estimates.idx_components_bad))
print('# Components remained: ' + str(cnm_obj.estimates.nr - len(cnm_obj.estimates.idx_components_bad)))
