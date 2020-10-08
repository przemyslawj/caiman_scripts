import itertools
import logging
import numpy as np

from caiman.base.rois import register_multisession, register_ROIs, com
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from matplotlib import pyplot as plt

from miniscope_file import gdrive_download_file
from load_args import *

logging.basicConfig(level=logging.INFO)

# Choose sessions
experiment_month = '2019-07'
exp_title_dates = {
    #'habituation': ['2019-08-27', '2019-08-28', '2019-08-29'],
    #'learning': ['2019-08-30', '2019-08-31', '2019-09-01']
    #'habituation': ['2019-08-29'],
    #'learning': ['2019-09-04', '2019-09-06']
    #'learning': ['2020-02-06', '2020-02-08', '2020-02-11']
    'learning': ['2019-07-28', '2019-07-30', '2019-08-01']
}

animal = 'B-BL'

filteredComponents = True
max_thr = 0.45
thresh_cost = 0.75
max_dist = 10

spatial = []
templates = []
cnm_list = []
cnm_titles = []
for exp_title in exp_title_dates.keys():
    exp_dates = exp_title_dates[exp_title]
    for exp_date in exp_dates:
        gdrive_dated_dir = os.path.join(downsample_subpath, experiment_month, exp_title, exp_date)
        local_dated_dir = os.path.join(local_rootdir, gdrive_dated_dir)
        result_dir = os.path.join(local_dated_dir, 'caiman', animal)
        gdrive_result_dir = os.path.join(gdrive_dated_dir, 'caiman', animal)

        analysis_results_fname = 'analysis_results.hdf5'
        if filteredComponents:
            analysis_results_fname = os.path.join('filtered', 'analysis_results_filtered.hdf5')
        h5fpath = os.path.join(result_dir, analysis_results_fname)
        if not os.path.isfile(h5fpath):
            gdrive_download_file(os.path.join(gdrive_result_dir, analysis_results_fname),
                                 os.path.dirname(h5fpath), rclone_config)
        cnm_obj = load_CNMF(h5fpath)
        cnm_list.append(cnm_obj)
        cnm_titles.append(exp_title + '_' + exp_date)
        spatial.append(cnm_obj.estimates.A.copy())

        rigid_template_fpath = result_dir + '/mc_rigid_template.npy'
        if not os.path.isfile(rigid_template_fpath):
            gdrive_download_file(gdrive_result_dir + '/mc_rigid_template.npy', result_dir, rclone_config)
        rigid_template = np.load(rigid_template_fpath)
        #templates.append(rigid_template)
        # Create a template using spatial footprints of the cells
        # Apply a threshold masks on spatial images
        A1 = np.stack([a * (a > max_thr * a.max()) for a in spatial[-1].toarray().T]).T
        # Calculate mean spatial footprint over all cells
        footprint_template = A1.mean(axis=1).reshape(cnm_obj.dims[::-1]).transpose()
        templates.append(footprint_template)
        dims = cnm_obj.dims


spatial_union, assignments, mappings = register_multisession(A=spatial, dims=dims, templates=templates,
                                                             thresh_cost=thresh_cost,
                                                             max_dist=max_dist,
                                                             max_thr=max_thr)


# Merge traces
# traces = np.zeros(assignments.shape, dtype=np.ndarray)
# for i in range(traces.shape[0]):
#     for j in range(traces.shape[1]):
#         if np.isnan(assignments[i,j]):
#             traces[i, j] = [np.nan] * cnm_list[j].estimates.C.shape[1]
#         else:
#             traces[i,j] = cnm_list[j].estimates.C[int(assignments[i,j])]

pairs = list(itertools.combinations(range(len(templates)), 2))
for pair in pairs:
    plt.figure()
    plt.suptitle(cnm_titles[pair[0]] + ' vs ' + cnm_titles[pair[1]])
    match_1, match_2, non_1, non_, perf_, A2 = register_ROIs(spatial[pair[0]], spatial[pair[1]], dims,
                                                            template1=templates[pair[0]],
                                                            template2=templates[pair[1]],
                                                            plot_results=True,
                                                            max_thr=max_thr,
                                                            thresh_cost=thresh_cost,
                                                            max_dist=max_dist,
                                                            Cn=templates[pair[0]])

    # Calculate centroid distances for the matched cells
    cm_1 = com(spatial[pair[0]], dims[0], dims[1])[match_1]
    cm_2 = com(spatial[pair[1]], dims[0], dims[1])[match_2]
    cm_2_registered = com(A2, dims[0], dims[1])[match_2]
    distances = [0] * len(cm_1)
    distances_registered = [0] * len(cm_1)
    for i, centroid1, centroid2, centroid2_reg in zip(range(len(cm_1)), cm_1, cm_2, cm_2_registered):
        distances[i] = np.linalg.norm(centroid1 - centroid2)
        distances_registered[i] = np.linalg.norm(centroid1 - centroid2_reg)
    print('Median distance=' + str(np.median(distances)))
    print('Median distance registered=' + str(np.median(distances_registered)))
