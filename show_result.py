from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf import params as params
import caiman as cm

import cv2
import numpy as np
import os

import video

from miniscope_file import gdrive_download_file, load_session_info

# Choose video
exp_month = '2019-08'
exp_title = 'habituation'
exp_date = '2019-08-28'
animal = 'F-BL'
rootdir = '/home/przemek/neurodata/'
gdrive_subdir = 'cheeseboard-down/down_2'


vid_index = 2
session_index = 1
reevaluate = False

rclone_config = os.environ['RCLONE_CONFIG']

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
session_lengths = np.cumsum([0] + session_info['session_lengths'])
session_trace_offset = session_lengths[session_index - 1]

dims = cnm_obj.estimates.dims
mmap_prefix = 'els'
mmap_fname = 'msCam' + str(vid_index) + '_' + mmap_prefix + '__d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_1_order_F_frames_1000_.mmap'
remote_mmap_dir = os.path.dirname(session_info['dat_files'][session_index - 1])
mmap_session_subdir = remote_mmap_dir.split(exp_date + '/')[1]
local_mmap_dir = os.path.join(local_dated_dir, mmap_session_subdir)
local_mmap_fpath = os.path.join(local_mmap_dir, mmap_fname)
if not os.path.isfile(local_mmap_fpath):
    gdrive_mmap_dir = '/'.join([gdrive_subdir, exp_month, exp_title, exp_date, mmap_session_subdir])
    gdrive_mmap_fpath = os.path.join(gdrive_mmap_dir, mmap_fname)
    gdrive_download_file(gdrive_mmap_fpath, local_mmap_dir, rclone_config)

# Load results

eval_params = {
    'cnn_lowest': .1,
    'min_cnn_thr': 0.9,
    'use_cnn': True,
    'rval_thr': 0.8,
    'rval_lowest': -1.0,
    'min_SNR': 6,
    'SNR_lowest': 2.5
}
opts = params.CNMFParams(params_dict=eval_params)
A = cnm_obj.estimates.A
frames = session_trace_offset + range((vid_index - 1) * 1000, vid_index * 1000)
images = video.load_images(local_mmap_fpath)
if reevaluate:
    print('evaluating components')
    from caiman.cluster import setup_cluster
    c, dview, n_processes = setup_cluster(backend='local', n_processes=None, single_thread=True)
    import miniscope_file
    all_images = video.load_images(miniscope_file.get_joined_memmap_fpath(result_dir))
    cnm_obj.estimates.threshold_spatial_components(maxthr=0.5, dview=dview)
    cnm_obj.estimates.remove_small_large_neurons(min_size_neuro=15, max_size_neuro=125)
    cnm_obj.estimates.evaluate_components(all_images, opts, dview)
    cm.stop_server(dview=dview)
else:
    eval_params['use_cnn'] = False
    cnm_obj.estimates.filter_components(images, cnm_obj.params, new_dict=eval_params)

print('Bad components: ' + str(cnm_obj.estimates.idx_components_bad))
print('# Components remained: ' + str(cnm_obj.estimates.nr - len(cnm_obj.estimates.idx_components_bad)))

""" Create movie with the cells activity """

# TODO: scale Y_res frame
def save_movie(estimate, imgs, Y_res, frames, q_max=99.5, q_min=2, gain=0.6,
               magnification=1,
               bpx=0, thr=0.,
               movie_name='results_movie.avi',
               fr=80,
               discard_bad_components=True):
    dims = imgs.shape[1:]
    if 'movie' not in str(type(imgs)):
        imgs = cm.movie(imgs)

    mov = imgs
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_vid_shape = [int(magnification * s) for s in mov.shape[1:][::-1]]
    out_vid_shape[0] *= 3 # Three vids are horizontally stacked
    out = cv2.VideoWriter(movie_name, fourcc, fr, tuple(out_vid_shape))
    cell_contours = dict()
    cell_idx = 0
    for a in estimate.A.T.toarray():
        a = a.reshape(dims, order='F')
        if bpx > 0:
            a = a[bpx:-bpx, bpx:-bpx]
        if magnification != 1:
            a = cv2.resize(a, None, fx=magnification, fy=magnification,
                           interpolation=cv2.INTER_LINEAR)
        ret, thresh = cv2.threshold(a, thr * np.max(a), 1., 0)
        contour, hierarchy = cv2.findContours(
            thresh.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        contours.append(contour)
        contours.append(list([c + np.array([[a.shape[1], 0]]) for c in contour]))
        contours.append(list([c + np.array([[2 * a.shape[1], 0]]) for c in contour]))
        cell_contours[cell_idx] = contours
        cell_idx += 1

    maxmov = np.nanpercentile(mov[0:10], q_max) if q_max < 100 else np.nanmax(mov)
    minmov = np.nanpercentile(mov[0:10], q_min) if q_min > 0 else np.nanmin(mov)
    index = 0
    F0 = np.reshape(estimate.b0, dims, order='F')
    components = range(estimate.A.shape[1])
    if discard_bad_components:
        components = estimate.idx_components
    A = estimate.A[:,components]
    for frame in mov:
        frame_index = frames[index]
        min_denoised_val = 30
        denoised_gain = 12
        denoised_frame = (np.reshape(A * estimate.C[components, frame_index], dims, order='F')) * denoised_gain + min_denoised_val
        denoised_frame = np.clip(denoised_frame, 0, 255)
        raw_frame = np.clip((frame - minmov) * 255. * gain / (maxmov - minmov), 0, 255)
        contours_frame = np.clip(4 * (frame - F0 * 0.7), 0, 255)
        if magnification != 1:
            raw_frame = cv2.resize(raw_frame, None, fx=magnification, fy=magnification,
                               interpolation=cv2.INTER_LINEAR)
            contours_frame = cv2.resize(contours_frame, None, fx=magnification, fy=magnification,
                                   interpolation=cv2.INTER_LINEAR)
            denoised_frame = cv2.resize(denoised_frame, None, fx=magnification, fy=magnification,
                               interpolation=cv2.INTER_LINEAR)
        res_frame = cv2.resize(Y_res[:,:,index], dsize=raw_frame.shape[::-1], interpolation=cv2.INTER_LINEAR)

        # Assume residuals should be normaly distributed and show only > 2 std
        res_frame_thr = np.where(res_frame > 1 * np.std(res_frame), res_frame + min_denoised_val / 5, 0)
        res_frame_thr = np.clip((res_frame_thr - np.min(res_frame_thr)) * denoised_gain / 2, 0, 255)
        denoised_frame = np.reshape(denoised_frame, denoised_frame.shape + (-1,))
        denoised_frame = np.concatenate([denoised_frame,
                                         denoised_frame,
                                         np.zeros_like(denoised_frame)],
                                        axis=2)

        denoised_frame[:, :, 2] = res_frame_thr
        rgbframe = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2RGB)

        contours_frame = cv2.cvtColor(contours_frame, cv2.COLOR_GRAY2RGB)
        for cell_idx in cell_contours.keys():
            yellow_col = (0, 255, 255)
            red_col = (0, 0, 255)
            contour_col = yellow_col
            if cell_idx in cnm_obj.estimates.idx_components_bad:
                contour_col = red_col
            for contour in cell_contours[cell_idx]:
                cv2.drawContours(contours_frame, contour, -1, contour_col, 1)

        concat_frame = np.concatenate([rgbframe,
                                       contours_frame,
                                       denoised_frame],
                                      axis=1)
        #cv2.imshow('frame', concat_frame.astype('uint8'))
        #if cv2.waitKey(30) & 0xFF == ord('q'):
        #    break
        out.write(concat_frame.astype('uint8'))
        index += 1

    out.release()


from cnmfe_model import model_residual
Y_res = model_residual(images, cnm_obj, 2, frames)

avifilename = 'Session' + str(session_index) + '_msCam' + str(vid_index) + '_result.avi'
save_movie(cnm_obj.estimates, images, Y_res, frames, q_max=75, magnification=2,
           bpx=0, thr=0.6, gain=0.4,
           movie_name=os.path.join(result_dir, avifilename))
