# Filters components based on the defined quality parameters and registers the components across sessions
from caiman.base.rois import register_multisession
from caiman.components_evaluation import evaluate_components_CNN
from caiman.utils.visualization import plot_contours
from matplotlib import pyplot as plt
import logging
import numpy as np
import subprocess

from results_format import save_matlab, readSFP
from miniscope_file import gdrive_download_file, gdrive_upload_file, load_session_info, load_hdf5_result, load_msresult
from load_args import *


logging.basicConfig(level=logging.INFO)

upload_results = True

# Relative to root paths to dated directories under which caiman results are stored
caiman_parent_paths = ['habituation', 'learning']

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
    'use_cnn': True,
    'rval_thr': 0.8,
    'rval_lowest': -1.0,
    'min_SNR': 6,
    #'min_SNR': 3,
    'SNR_lowest': 2.5,
    'min_cnn_thr': 0.99,
    'cnn_lowest': 0.1
}

registration_params = {
    'max_thr': 0.45,
    'thresh_cost': 0.75,
    'max_dist': 10
}


def filter_components(cnm_obj, components_quality_params, registration_params, exp_date):
    cnm_obj.estimates.threshold_spatial_components(maxthr=registration_params['max_thr'])
    cnm_obj.estimates.remove_small_large_neurons(min_size_neuro=neuron_size_params['min'],
                                                 max_size_neuro=neuron_size_params['max'])
    logging.info('%s: filtered %d neurons based on the size', exp_date, len(cnm_obj.estimates.idx_components_bad))
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

    logging.info(exp_date + ', bad components: ' + str(cnm_obj.estimates.idx_components_bad))
    logging.info(exp_date + ', # Components remained: ' + str(cnm_obj.estimates.nr - len(cnm_obj.estimates.idx_components_bad)))

    cnm_obj.estimates.select_components(use_object=True)
    return cnm_obj


# Load session objects
session_objs = []
for exp_title in caiman_parent_paths:
    gdrive_exp_dir = os.path.join(upload_path, exp_title)
    cp = subprocess.run(['rclone', 'lsf', '--config', 'env/rclone.conf', rclone_config + ':' + gdrive_exp_dir],
                        capture_output=True, text=True)
    exp_dates = [x.strip() for x in cp.stdout.split('\n') if len(x.strip()) > 0]
    for exp_date in exp_dates:
        gdrive_dated_dir = os.path.join(gdrive_exp_dir, exp_date)
        gdrive_result_dir = os.path.join(gdrive_dated_dir, 'caiman', animal_name)
        local_dated_dir = os.path.join(local_rootdir, downsample_subpath, exp_title, exp_date)
        result_dir = os.path.join(local_dated_dir, 'caiman', animal_name)

        cnm_obj = load_hdf5_result(result_dir, gdrive_result_dir, rclone_config)
        if cnm_obj is None:
            print('HDF5 file not found, skipping exp_date=' + exp_date)
            continue

        rigid_template_fpath = result_dir + '/mc_rigid_template.npy'
        if not os.path.isfile(rigid_template_fpath):
            success = gdrive_download_file(gdrive_result_dir + '/mc_rigid_template.npy', result_dir, rclone_config)
            if not success:
                print('Template file not found, skipping exp_date=' + exp_date)
                continue

        # Create a template using spatial footprints of the cells
        # Apply a threshold masks on spatial images
        A = cnm_obj.estimates.A
        A1 = np.stack([a * (a > registration_params['max_thr'] * a.max()) for a in A.toarray().T]).T
        # Calculate mean spatial footprint over all cells
        footprint_template = A1.mean(axis=1).reshape(cnm_obj.dims[::-1]).transpose()

        session_objs.append({
            'cnm_obj': cnm_obj,
            'session_info': load_session_info(result_dir, gdrive_result_dir, rclone_config),
            'template': footprint_template,
            'gdrive_result_dir': gdrive_result_dir,
            'result_dir': result_dir,
            'exp_date': exp_date
        })


# Filter components
for session_i, session in enumerate(session_objs):
    logging.info('Filtering session from ' + session['exp_date'])
    session['cnm_obj_filtered'] = filter_components(session['cnm_obj'],
                                                    components_quality_params,
                                                    registration_params,
                                                    session['exp_date'])

# Do multisession registration
spatial_comps = [d['cnm_obj_filtered'].estimates.A for d in session_objs]
templates = [d['template'] for d in session_objs]

spatial_union, assignments, mappings = register_multisession(A=spatial_comps,
                                                             dims=session_objs[0]['cnm_obj_filtered'].dims,
                                                             templates=templates,
                                                             thresh_cost=registration_params['thresh_cost'],
                                                             max_dist=registration_params['max_dist'],
                                                             max_thr=registration_params['max_thr'],
                                                             use_opt_flow=True)


# Save updated cnm_obj and upload to drive
for sess_i, session in enumerate(session_objs):
    session_info = session['session_info']
    cnm_obj = session['cnm_obj_filtered']
    cnm_obj.estimates.registered_cell_ids = mappings[sess_i]
    import subprocess
    subprocess.run(['mkdir', '-p', os.path.join(session['result_dir'], 'filtered')])
    cnm_obj_fpath = os.path.join(session['result_dir'], 'filtered', 'analysis_results_filtered.hdf5')
    cnm_obj.save(cnm_obj_fpath)
    gdrive_upload_dir = os.path.join(
        session['result_dir'][session['result_dir'].find(downsample_subpath):],
        'filtered')
    if upload_results:
        gdrive_upload_file(cnm_obj_fpath, gdrive_upload_dir, rclone_config)

    # Spatial footprint of filtered components
    plt.figure()
    plot_contours(cnm_obj.estimates.A, np.zeros(cnm_obj.dims), thr=0.9)
    plt.imshow(np.amax(readSFP(cnm_obj, False), axis=2))
    plt.savefig(os.path.join(session['result_dir'], 'filtered', 'sfp.svg'), edgecolor='w',
                format='svg', transparent=False)

    msresult = load_msresult(session['result_dir'], session['gdrive_result_dir'], rclone_config)
    msobj = msresult['ms'][0, 0]
    ms_fpath, sfp_fpath = save_matlab(cnm_obj, session_info, os.path.join(session['result_dir'], 'filtered'), [],
                                      msobj['time'], msobj['camNumber'],
                                      extraFields={'cellId': mappings[sess_i]})
    if upload_results:
        gdrive_upload_file(ms_fpath, gdrive_upload_dir, rclone_config)
        gdrive_upload_file(sfp_fpath, gdrive_upload_dir, rclone_config)

