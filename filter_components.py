from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.components_evaluation import evaluate_components_CNN
from miniscope_file import gdrive_download_file
from load_args import *
import results_format

from matplotlib import pyplot as plt
import caiman as cm
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# Choose video
exp_month = '2020-01'
exp_title = 'learning'
exp_date = '2020-01-30'

#rootdir = '/home/przemek/neurodata/'
rootdir = '/mnt/DATA/Prez/'
gdrive_subdir = 'cheeseboard-down/down_2'

vca1_neuron_sizes = {
    'max': 250,
    'min': 40
}
dca1_neuron_sizes = {
    'max': 140,
    'min': 10
}
neuron_size_params = vca1_neuron_sizes

components_quality_params = {
    'use_cnn': False,
    'rval_thr': 0.8,
    'rval_lowest': -1.0,
    'min_SNR': 6,
    #'min_SNR': 3,
    'SNR_lowest': 2.5,
    'min_cnn_thr': 0.99,
    'cnn_lowest': 0.05
}

registration_params = {
    'max_thr': 0.45,
    'thresh_cost': 0.75,
    'max_dist': 10
}


""" Prepare data """
gdrive_dated_dir = os.path.join(gdrive_subdir, exp_month, exp_title, exp_date)
gdrive_result_dir = os.path.join(gdrive_dated_dir, 'caiman', animal_name)

caiman_result_dir = os.path.join(local_miniscope_path, exp_title, exp_date, 'caiman', animal_name)
# Download result files if not stored locally
h5fpath = os.path.join(caiman_result_dir, 'analysis_results.hdf5')
if not os.path.isfile(h5fpath):
    gdrive_download_file(gdrive_result_dir + '/analysis_results.hdf5', caiman_result_dir, rclone_config)
cnm_obj = load_CNMF(h5fpath)


def filter_components(cnm_obj, components_quality_params, registration_params):
    cnm_obj.estimates.threshold_spatial_components(maxthr=registration_params['max_thr'])
    cnm_obj.estimates.remove_small_large_neurons(min_size_neuro=neuron_size_params['min'],
                                                 max_size_neuro=neuron_size_params['max'])
    logging.info('filtered %d neurons based on the size', len(cnm_obj.estimates.idx_components_bad))
    cnm_obj.estimates.accepted_list = cnm_obj.estimates.idx_components
    empty_images = np.zeros((1,) + cnm_obj.dims)
    if components_quality_params['use_cnn'] and (
            not hasattr(cnm_obj.estimates, 'cnn_preds') or len(cnm_obj.estimates.cnn_preds) == 0):
        predictions, final_crops = evaluate_components_CNN(
            cnm_obj.estimates.A, cnm_obj.dims, cnm_obj.params.init['gSig'],
            model_name=os.path.join(caiman_src_datadir, 'model', 'cnn_model'))
        cnm_obj.estimates.cnn_preds = predictions[:, 1]
    cnm_obj.estimates.filter_components(empty_images, cnm_obj.params,
                                        new_dict=components_quality_params,
                                        select_mode='Accepted')

    logging.info('Bad components: ' + str(cnm_obj.estimates.idx_components_bad))
    logging.info('# Components remained: ' + str(cnm_obj.estimates.nr - len(cnm_obj.estimates.idx_components_bad)))

    cnm_obj.estimates.select_components(use_object=True)
    return cnm_obj


filtered_obj = filter_components(cnm_obj, components_quality_params, registration_params)

plt.figure()
cm.utils.visualization.plot_contours(filtered_obj.estimates.A, np.zeros(cnm_obj.dims), thr=0.9)
plt.imshow(np.amax(results_format.readSFP(filtered_obj, False), axis=2))



