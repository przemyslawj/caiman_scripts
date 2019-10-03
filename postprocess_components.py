# Filters components based on the defined quality parameters and registers the components across sessions
from caiman.base.rois import register_multisession, register_ROIs, com
import logging
import numpy as np
import os
import subprocess
from scipy.ndimage import center_of_mass

from miniscope_file import gdrive_download_file, load_session_info, load_hdf5_result
logging.basicConfig(level=logging.INFO)

# Choose sessions
rootdir = '/home/przemek/neurodata/'
gdrive_subdir = 'cheeseboard-down/down_2'
exp_month = '2019-08'
exp_titles = ['habituation']
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
    foo_images = np.zeros((1,) + (cnm_obj.dims))
    cnm_obj.estimates.filter_components(foo_images, cnm_obj.params,
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


def find_centroids(SFP):
    return [center_of_mass(SFP[:, :, ii]) for ii in range(SFP.shape[2])]


def get_timestamps(session_info):
    mstime = np.array([], dtype=np.int)
    i = 0
    for dat_file in session_info['dat_files']:
        gdrive_dat_fpath = dat_file[dat_file.find(gdrive_subdir):]
        local_dat_fpath = os.path.join(rootdir, gdrive_dat_fpath)
        if not os.path.isfile(local_dat_fpath):
            gdrive_download_file(gdrive_dat_fpath, os.path.dirname(local_dat_fpath), rclone_config)
        with open(local_dat_fpath) as f:
            camNum, frameNum, sysClock, buffer = np.loadtxt(f, dtype='float', comments='#', skiprows=1, unpack=True)
        camNumber = camNum[0]
        mstime_idx = np.where(camNum == camNumber)
        this_mstime = sysClock[mstime_idx]
        this_mstime = this_mstime[0:session_info['session_lengths'][i]]
        mstime = np.concatenate([mstime, this_mstime])
        i += 1

    mstime[0] = 0
    return mstime, camNumber


def save_ms_file(result_data_dir, cnm_obj, session_info, mstime, camNumber, unionSFP):
    from scipy.io import savemat
    nneurons = cnm_obj.estimates.nr
    SFP = cnm_obj.estimates.A.toarray().reshape(cnm_obj.dims + (nneurons,), order='F')
    results_dict = {
        'Experiment': animal,
        'numFiles': 1,
        'framesNum': cnm_obj.estimates.C.shape[1],
        'maxFramesPerFile': 1000,
        'height': cnm_obj.dims[0],
        'width': cnm_obj.dims[1],
        'camNumber': camNumber,
        'time': mstime,
        'sessionLengths': session_info['session_lengths'],
        #'meanFrame': meanFrame,
        'Centroids': find_centroids(SFP),
        #'CorrProj': cn_filter,
        #'PeakToNoiseProj': pnr,
        #'PNR': cnm.estimates.SNR_comp,  # Calculated Peak-to-Noise ratios
        'neurons_sn': cnm_obj.estimates.neurons_sn,  # neurons noise estimation
        'RawTraces': cnm_obj.estimates.C.conj().transpose(),  # swap time x neurons dimensions
        'DeconvTraces': cnm_obj.estimates.S.conj().transpose(),
        'SFPs': SFP,
        'unionSFP': unionSFP,
        'registered_cell_id': cnm_obj.estimates.registered_cell_ids,
        'numNeurons': nneurons,
    }

    SFPperm = np.transpose(SFP, [2, 0, 1])
    savemat(result_data_dir + '/SFP.mat', {'SFP': SFPperm})
    savemat(result_data_dir + '/ms.mat', {'ms': results_dict})



for sess_i, session in enumerate(session_objs):
    session_info = session['session_info']
    mstime, camNumber = get_timestamps(session_info)
    cnm_obj.estimates.registered_cell_ids = mappings[sess_i]
    cnm_obj.save(result_dir + '/analysis_result_filtered.hdf5')

    unionSFP = spatial_union.reshape(cnm_obj.dims + (-1,), order='F')
    save_ms_file(session['result_dir'], cnm_obj, session_info, mstime, camNumber, unionSFP)
