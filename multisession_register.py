import itertools
import logging
import numpy as np
import pandas as pd

from caiman.base.rois import register_multisession, register_ROIs, com
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from matplotlib import pyplot as plt

from miniscope_file import gdrive_download_file
from load_args import *

logging.basicConfig(level=logging.INFO)

# Choose sessions
exp_title_dates = {
    #'habituation': ['2019-08-27', '2019-08-28', '2019-08-29'],
    'learning': ['2019-07-28', '2019-07-30']
    #'habituation': ['2019-08-29'],
    #'learning': ['2019-09-04', '2019-09-06', '2019-09-08']
    #'learning': ['2020-02-06', '2020-02-08', '2020-02-11']
    #'habituation': ['2020_10_08'],
    #'learning': ['2020_10_24', '2020_10_22', '2020_10_20', '2020_10_18']
}
export_stats = False

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
        local_dated_dir = os.path.join(local_rootdir, downsample_subpath, exp_title, exp_date)
        gdrive_result_dir = os.path.join(upload_path, exp_title, exp_date, 'caiman', animal_name)

        analysis_results_fname = 'analysis_results.hdf5'
        if filteredComponents:
            analysis_results_fname = os.path.join('filtered', 'analysis_results_filtered.hdf5')
        caiman_result_dir = os.path.join(local_miniscope_path, exp_title, exp_date, 'caiman', animal_name)
        h5fpath = os.path.join(caiman_result_dir, analysis_results_fname)
        if not os.path.isfile(h5fpath):
            gdrive_download_file(os.path.join(gdrive_result_dir, analysis_results_fname),
                                 os.path.dirname(h5fpath), rclone_config)
        cnm_obj = load_CNMF(h5fpath)
        cnm_list.append(cnm_obj)
        cnm_titles.append(exp_title + '_' + exp_date)
        spatial.append(cnm_obj.estimates.A.copy())

        rigid_template_fpath = os.path.join(caiman_result_dir, 'mc_rigid_template.npy')
        if not os.path.isfile(rigid_template_fpath):
            gdrive_download_file(gdrive_result_dir + '/mc_rigid_template.npy', caiman_result_dir, rclone_config)
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

def my_roi_image(A1,
                 A2,
                 dims,
                 matched_ROIs1,
                 matched_ROIs2,
                 non_matched1,
                 non_matched2,
                 max_thr):
    if 'ndarray' not in str(type(A1)):
        A1 = A1.toarray()
    if 'ndarray' not in str(type(A2)):
        A2 = A2.toarray()

    def array2img(A):
        A_thr = np.stack([a * (a > max_thr * a.max()) for a in A.T]).T
        lp, hp = np.nanpercentile(A_thr.sum(1), [5, 99])
        return (np.reshape(A_thr.sum(1), dims, order='F') - lp) / (hp - lp)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    img1 = array2img(A1)
    img2 = array2img(A2)
    zero_img = np.zeros_like(img1)
    plt.imshow(np.dstack((img1, img2, zero_img)))
    plt.title('All')

    plt.subplot(1, 3, 2)
    img1 = array2img(A1[:, matched_ROIs1])
    img2 = array2img(A2[:, matched_ROIs2])
    plt.imshow(np.dstack((img1, img2, zero_img)))
    plt.imshow(np.dstack((img1, img2, zero_img)))
    plt.title('Matched')

    plt.subplot(1, 3, 3)
    img1 = array2img(A1[:, non_matched1])
    img2 = array2img(A2[:, non_matched2])
    plt.imshow(np.dstack((img1, img2, zero_img)))
    plt.imshow(np.dstack((img1, img2, zero_img)))
    plt.title('Not matched')


def footprint_cors(A1, A2, centroids, window_px=10):
    if 'ndarray' not in str(type(A1)):
        A1 = A1.toarray()
    if 'ndarray' not in str(type(A2)):
        A2 = A2.toarray()

    def crop_vector(A, centroid):
        a = np.reshape(A, dims, order='F')
        centroid = [int(x) for x in centroid]

        a_cropped = a[max(0, centroid[0] - window_px):min(centroid[0] + window_px, dims[0]),
                      max(0, centroid[1] - window_px):min(centroid[1] + window_px, dims[1])]
        return np.reshape(a_cropped, (a_cropped.shape[0] * a_cropped.shape[1],), order='F')

    cors = np.zeros((A1.shape[1]), dtype='float32')
    for cell_i in range(A1.shape[1]):
        #a = crop_vector(A1[:, cell_i], centroids[cell_i, :])
        #v = crop_vector(A2[:, cell_i], centroids[cell_i, :])
        #cors[cell_i] = np.correlate(a, v)
        cors[cell_i] = np.correlate(A1[:, cell_i], A2[:, cell_i])
    return cors


pairs = list(itertools.combinations(range(len(templates)), 2))

stats = pd.DataFrame()

for pair in pairs:
    match_1, match_2, non_1, non_2, perf_, A2 = register_ROIs(spatial[pair[0]], spatial[pair[1]], dims,
                                                            template1=templates[pair[0]],
                                                            template2=templates[pair[1]],
                                                            plot_results=False,
                                                            max_thr=max_thr,
                                                            thresh_cost=thresh_cost,
                                                            max_dist=max_dist,
                                                            Cn=templates[pair[0]])
    plt.suptitle(cnm_titles[pair[0]] + ' vs ' + cnm_titles[pair[1]])

    my_roi_image(spatial[pair[0]].toarray(), A2, dims, match_1, match_2, non_1, non_2, max_thr)

    # Calculate centroid distances for the matched cells
    cm_1 = com(spatial[pair[0]], dims[0], dims[1])[match_1]
    cm_2 = com(spatial[pair[1]], dims[0], dims[1])[match_2]
    cm_2_registered = com(A2, dims[0], dims[1])[match_2]
    distances = [0] * len(cm_1)
    distances_registered = [0] * len(cm_1)
    matched_cors = footprint_cors(spatial[pair[0]][:, match_1],
                                  A2[:, match_2],
                                  cm_1)
    for i, centroid1, centroid2, centroid2_reg in zip(range(len(cm_1)), cm_1, cm_2, cm_2_registered):
        distances[i] = np.linalg.norm(centroid1 - centroid2)
        distances_registered[i] = np.linalg.norm(centroid1 - centroid2_reg)
    print('Median distance=' + str(np.median(distances)))
    print('Median distance registered=' + str(np.median(distances_registered)))
    print('Median correlation of matches=' + str(np.median(matched_cors)))
    trial_stats = pd.DataFrame()
    trial_stats['reg_dist_px'] = distances_registered
    trial_stats['matched_cors'] = matched_cors
    trial_stats['exp_day1'] = cnm_titles[pair[0]]
    trial_stats['exp_day2'] = cnm_titles[pair[1]]
    stats = stats.append(trial_stats)


if export_stats:
    stats.to_csv(os.path.join('/home/prez/tmp/cheeseboard/registration_stats', animal_name + '.csv'))