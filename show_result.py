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
exp_date = '2019-08-27'
animal = 'F-TL'
rootdir = '/home/przemek/neurodata/'
gdrive_subdir = 'cheeseboard-down/down_2'


vid_index = 2
session_index = 1

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
print('loading images')
images = video.load_images(local_mmap_fpath)

eval_params = {
    'cnn_lowest': .1,
    'min_cnn_thr': 0.95,
    'rval_thr': 0.85,
    'rval_lowest': 0.5,
    'min_SNR': 5,
    'SNR_lowest': 2
}
opts = params.CNMFParams(params_dict=eval_params)
#print('evaluating components')
#cnm_obj.estimates.evaluate_components(images, opts, dview)
print('Bad components: ' + str(cnm_obj.estimates.idx_components_bad))
print('# Components remained: ' + str(cnm_obj.estimates.nr - len(cnm_obj.estimates.idx_components_bad)))

""" Create movie with the cells activity """


def save_movie(estimate, imgs, frames, q_max=99.5, q_min=2, gain=0.6,
               magnification=1,
               bpx=0, thr=0.,
               movie_name='results_movie.avi',
               fr=80):
    dims = imgs.shape[1:]
    if 'movie' not in str(type(imgs)):
        imgs = cm.movie(imgs)

    mov = imgs
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_vid_shape = [int(magnification * s) for s in mov.shape[1:][::-1]]
    out_vid_shape[0] *= 3 # Three vids are horizontally stacked
    out = cv2.VideoWriter(movie_name, fourcc, fr, tuple(out_vid_shape))
    contours = []
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
        contours.append(contour)
        contours.append(list([c + np.array([[a.shape[1], 0]]) for c in contour]))
        contours.append(list([c + np.array([[2 * a.shape[1], 0]]) for c in contour]))

    maxmov = np.nanpercentile(mov[0:10], q_max) if q_max < 100 else np.nanmax(mov)
    minmov = np.nanpercentile(mov[0:10], q_min) if q_min > 0 else np.nanmin(mov)
    index = 0
    F0 = np.reshape(estimate.b0, dims[::-1]).T
    for frame in mov:
        frame_index = frames[index]
        min_denoised_val = 20
        denoised_gain = 6
        denoised_frame = (np.reshape(estimate.A * estimate.C[:, frame_index], dims[::-1])) * denoised_gain + min_denoised_val
        denoised_frame = np.clip(denoised_frame, 0, 255)
        denoised_frame = denoised_frame.T
        raw_frame = np.clip((frame - minmov) * 255. * gain / (maxmov - minmov), 0, 255)
        #contours_frame = np.clip((frame - F0) / F0 * 255 + 20, 0, 255)
        contours_frame = np.clip(4 * (frame - F0 * 0.7), 0, 255)
        if magnification != 1:
            raw_frame = cv2.resize(raw_frame, None, fx=magnification, fy=magnification,
                               interpolation=cv2.INTER_LINEAR)
            contours_frame = cv2.resize(contours_frame, None, fx=magnification, fy=magnification,
                                   interpolation=cv2.INTER_LINEAR)
            denoised_frame = cv2.resize(denoised_frame, None, fx=magnification, fy=magnification,
                               interpolation=cv2.INTER_LINEAR)

        rgbframe = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2RGB)
        #contours_frame = np.copy(raw_frame)
        contours_frame = cv2.cvtColor(contours_frame, cv2.COLOR_GRAY2RGB)
        for contour in contours:
            cv2.drawContours(contours_frame, contour, -1, (0, 255, 255), 1)
        concat_frame = np.concatenate([rgbframe,
                                       contours_frame,
                                       cv2.cvtColor(denoised_frame.astype('float32'), cv2.COLOR_GRAY2RGB)],
                                      axis=1)
        #cv2.imshow('frame', concat_frame.astype('uint8'))
        #if cv2.waitKey(30) & 0xFF == ord('q'):
        #    break
        out.write(concat_frame.astype('uint8'))
        index += 1

    out.release()


cnm_obj.estimates.W = None
A = cnm_obj.estimates.A
frames = session_trace_offset + range((vid_index - 1) * 1000, vid_index * 1000)

avifilename = 'Session' + str(session_index) + '_msCam' + str(vid_index) + '_result.avi'
save_movie(cnm_obj.estimates, images, frames, q_max=75, magnification=2,
           bpx=0, thr=0.6, gain=0.4,
           movie_name=os.path.join(result_dir, avifilename))
