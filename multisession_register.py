import itertools
import os
from caiman.base.rois import register_multisession, register_ROIs, com
from matplotlib import pyplot as plt
import numpy as np

from miniscope_file import gdrive_download_file, load_session_info, load_hdf5_result

# Choose sessions
exp_month = '2019-08'
exp_title = 'habituation'
exp_dates = ['2019-08-27', '2019-08-28', '2019-08-29']
animal = 'E-BL'
rootdir = '/home/przemek/neurodata/'
gdrive_subdir = 'cheeseboard-down/down_2'

rclone_config = os.environ['RCLONE_CONFIG']

spatial = []
templates = []
cnm_list = []
for exp_date in exp_dates:
    gdrive_dated_dir = os.path.join(gdrive_subdir, exp_month, exp_title, exp_date)
    local_dated_dir = os.path.join(rootdir, gdrive_dated_dir)
    result_dir = os.path.join(local_dated_dir, 'caiman', animal)
    gdrive_result_dir = os.path.join(gdrive_dated_dir, 'caiman', animal)

    cnm_obj = load_hdf5_result(result_dir, gdrive_result_dir, rclone_config)
    cnm_list.append(cnm_obj)
    spatial.append(cnm_obj.estimates.A.copy())

    rigid_template_fpath = result_dir + '/mc_rigid_template.npy'
    if not os.path.isfile(rigid_template_fpath):
        gdrive_download_file(gdrive_result_dir + '/mc_rigid_template.npy', result_dir, rclone_config)
    templates.append(np.load(rigid_template_fpath))
    dims = cnm_obj.dims

max_thr = 0.5
thresh_cost = 0.75
max_dist = 10
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
    match_1, match_2, non_1, non_, perf_, A2 = register_ROIs(spatial[pair[0]], spatial[pair[1]], dims,
                                                            template1=templates[pair[0]],
                                                            template2=templates[pair[1]],
                                                            plot_results=True,
                                                            max_thr=max_thr,
                                                            thresh_cost=thresh_cost,
                                                            max_dist=max_dist,
                                                            Cn=templates[-1])

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
