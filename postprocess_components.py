# Filters components based on the defined quality parameters and registers the components across sessions
from caiman.base.rois import register_multisession, register_ROIs, com
import logging
import numpy as np
import os
import subprocess

from results_format import save_matlab
from miniscope_file import gdrive_download_file, gdrive_upload_file, load_session_info, load_hdf5_result, load_msresult
logging.basicConfig(level=logging.INFO)

# Choose sessions
rootdir = '/home/przemek/neurodata/'
gdrive_subdir = 'cheeseboard-down/down_2'
exp_month = '2019-08'
exp_titles = ['habituation', 'learning']
animal = 'E-BL'
rclone_config = os.environ['RCLONE_CONFIG']

components_quality_params = {
    'use_cnn': False,
    'rval_thr': 0.8,
    'rval_lowest': -1.0,
    'min_SNR': 6,
    'SNR_lowest': 2.5
}


def filter_components(cnm_obj, components_quality_params):
    cnm_obj.estimates.threshold_spatial_components(maxthr=0.5)
    cnm_obj.estimates.remove_small_large_neurons(min_size_neuro=20, max_size_neuro=110)
    cnm_obj.estimates.accepted_list = cnm_obj.estimates.idx_components
    empty_images = np.zeros((1,) + (cnm_obj.dims))
    cnm_obj.estimates.filter_components(empty_images, cnm_obj.params,
                                        new_dict=components_quality_params,
                                        select_mode='Accepted')

    print(exp_date + ', bad components: ' + str(cnm_obj.estimates.idx_components_bad))
    print(exp_date + ', # Components remained: ' + str(cnm_obj.estimates.nr - len(cnm_obj.estimates.idx_components_bad)))

    cnm_obj.estimates.select_components(use_object=True)
    return cnm_obj


# Load session objects
session_objs = []
for exp_title in exp_titles:
    gdrive_exp_dir = os.path.join(gdrive_subdir, exp_month, exp_title)
    cp = subprocess.run(['rclone', 'lsf', '--config', 'env/rclone.conf', rclone_config + ':' + gdrive_exp_dir],
                         capture_output=True, text=True)
    exp_dates = [x.strip() for x in cp.stdout.split('\n') if len(x.strip()) > 0]
    for exp_date in exp_dates:
        gdrive_dated_dir = os.path.join(gdrive_exp_dir, exp_date)
        gdrive_result_dir = os.path.join(gdrive_dated_dir, 'caiman', animal)
        local_dated_dir = os.path.join(rootdir, gdrive_dated_dir)
        result_dir = os.path.join(local_dated_dir, 'caiman', animal)

        cnm_obj = load_hdf5_result(result_dir, gdrive_result_dir, rclone_config)

        rigid_template_fpath = result_dir + '/mc_rigid_template.npy'
        if not os.path.isfile(rigid_template_fpath):
            gdrive_download_file(gdrive_result_dir + '/mc_rigid_template.npy', result_dir, rclone_config)

        session_objs.append({
            'cnm_obj': cnm_obj,
            'session_info': load_session_info(result_dir, gdrive_result_dir, rclone_config),
            'template': np.load(rigid_template_fpath),
            'gdrive_result_dir': gdrive_result_dir,
            'result_dir': result_dir
        })


# Filter components
for session in session_objs:
    session['cnm_obj'] = filter_components(session['cnm_obj'], components_quality_params)

# Do multisession registration
spatial_comps = [d['cnm_obj'].estimates.A for d in session_objs]
templates = [d['template'] for d in session_objs]

max_thr = 0.5
thresh_cost = 0.75
max_dist = 10
spatial_union, assignments, mappings = register_multisession(A=spatial_comps,
                                                             dims=session_objs[0]['cnm_obj'].dims,
                                                             templates=templates,
                                                             thresh_cost=thresh_cost,
                                                             max_dist=max_dist,
                                                             max_thr=max_thr)


# Save updated cnm_obj and upload to drive
for sess_i, session in enumerate(session_objs):
    session_info = session['session_info']
    cnm_obj = session['cnm_obj']
    cnm_obj.estimates.registered_cell_ids = mappings[sess_i]
    import subprocess
    subprocess.run(['mkdir', '-p', os.path.join(session['result_dir'], 'filtered')])
    cnm_obj_fpath = os.path.join(session['result_dir'], 'filtered', 'analysis_results_filtered.hdf5')
    cnm_obj.save(cnm_obj_fpath)
    gdrive_upload_dir = os.path.join(
        session['result_dir'][session['result_dir'].find(gdrive_subdir):],
        'filtered')
    gdrive_upload_file(cnm_obj_fpath, gdrive_upload_dir, rclone_config)

    msresult = load_msresult(session['result_dir'], session['gdrive_result_dir'], rclone_config)
    ms_fpath, sfp_fpath = save_matlab(cnm_obj, session_info, os.path.join(session['result_dir'], 'filtered'), [],
                                      msresult['ms'][0,0]['time'], msresult['ms'][0,0]['camNumber'],
                                      extraFields={'cellId': mappings[sess_i]})
    gdrive_upload_file(ms_fpath, gdrive_upload_dir, rclone_config)
    gdrive_upload_file(sfp_fpath, gdrive_upload_dir, rclone_config)
