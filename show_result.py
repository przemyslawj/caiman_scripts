from caiman.source_extraction.cnmf import params as params
import caiman as cm

import cv2
import numpy as np

import video
import logging

logging.basicConfig(level=logging.INFO)

from miniscope_file import gdrive_download_file, load_session_info, load_hdf5_result
from load_args import *

# Choose video
trial_no = int(os.environ['TRIAL_NO'])

vid_start_index = 0
vid_index = 1
reevaluate = False
filtered = False


""" Prepare data """
gdrive_dated_dir = os.path.join(upload_path, experiment_title, experiment_date)
gdrive_result_dir = os.path.join(gdrive_dated_dir, 'caiman', animal_name)
# Download result files if not stored locally
cnm_obj = load_hdf5_result(caiman_result_dir, gdrive_result_dir, rclone_config, use_filtered=filtered)

session_info = load_session_info(caiman_result_dir, gdrive_result_dir, rclone_config)
session_lengths = np.cumsum([0] + session_info['session_lengths'])
session_trace_offset = session_lengths[trial_no - 1]

dims = cnm_obj.estimates.dims
mmap_prefix = 'els'
mmap_fname = vid_prefix + str(vid_index) + '_' + mmap_prefix + '__d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_1_order_F_frames_1000_.mmap'
remote_mmap_dir = os.path.dirname(session_info['timestamp_files'][trial_no - 1])
mmap_session_subdir = remote_mmap_dir.split(experiment_date + '/')[1]
local_mmap_dir = os.path.join(local_miniscope_path, experiment_title, experiment_date, mmap_session_subdir)
local_mmap_fpath = os.path.join(local_mmap_dir, mmap_fname)
if not os.path.isfile(local_mmap_fpath):
    gdrive_mmap_dir = '/'.join([upload_path, experiment_title, experiment_date, mmap_session_subdir])
    gdrive_mmap_fpath = os.path.join(gdrive_mmap_dir, mmap_fname)
    gdrive_download_file(gdrive_mmap_fpath, local_mmap_dir, rclone_config)

# Load results

eval_params = {
    'use_cnn': False,
    'rval_thr': 0.8,
    'rval_lowest': -1.0,
    #'min_SNR': 8,
    #'SNR_lowest': 2.5,
    'min_SNR': 6,
    'SNR_lowest': 2.5,
}
max_thr = 0.45

vca1_neuron_sizes = {
    'max': 200,
    'min': 10
}
dca1_neuron_sizes = {
    'max': 110,
    #'min': 20
    'min': 5
}
neuron_size_params = vca1_neuron_sizes

opts = params.CNMFParams(params_dict=eval_params)
A = cnm_obj.estimates.A
frames = session_trace_offset + range((vid_index - vid_start_index) * 1000, (vid_index - vid_start_index + 1) * 1000)
images = video.load_images(local_mmap_fpath)
if not filtered:
    cnm_obj.estimates.threshold_spatial_components(maxthr=max_thr)
    cnm_obj.estimates.remove_small_large_neurons(min_size_neuro=neuron_size_params['min'],
                                                max_size_neuro=neuron_size_params['max'])

idx_components_bad = cnm_obj.estimates.idx_components_bad
if idx_components_bad is None:
    idx_components_bad = []

if reevaluate:
    print('evaluating components')
    from caiman.cluster import setup_cluster
    c, dview, n_processes = setup_cluster(backend='local', n_processes=None, single_thread=True)
    import miniscope_file
    all_images = video.load_images(miniscope_file.get_joined_memmap_fpath(caiman_result_dir))
    cnm_obj.estimates.evaluate_components(all_images, opts, dview)
    cm.stop_server(dview=dview)
else:
    eval_params['use_cnn'] = False
    #cnm_obj.estimates.filter_components(images, cnm_obj.params, new_dict=eval_params)
cnm_obj.estimates.idx_components_bad = sorted(np.union1d(cnm_obj.estimates.idx_components_bad, idx_components_bad))
cnm_obj.estimates.idx_components = sorted(np.setdiff1d(np.arange(cnm_obj.estimates.A.shape[-1]), cnm_obj.estimates.idx_components_bad))

print('Bad components: ' + str(cnm_obj.estimates.idx_components_bad))
print('# Components remained: ' + str(cnm_obj.estimates.nr - len(cnm_obj.estimates.idx_components_bad)))

""" Create movie with the cells activity """

# TODO: scale Y_res frame
def save_movie(estimate, imgs, Y_res, B, frames, q_max=99.5, q_min=2, bpx=0, thr=0.6, gain=0.6,
               magnification=1,
               movie_name='results_movie.avi',
               fr=60,
               discard_bad_components=True):
    dims = imgs.shape[1:]
    if 'movie' not in str(type(imgs)):
        imgs = cm.movie(imgs)

    mov = imgs
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_vid_shape = [int(magnification * s) for s in mov.shape[1:][::-1]]
    out_vid_shape[0] *= 3 # Three vids are horizontally stacked
    out = cv2.VideoWriter(movie_name, fourcc, fr, tuple(out_vid_shape))


    maxmov = np.nanpercentile(mov[0:10], q_max) if q_max < 100 else np.nanmax(mov)
    minmov = np.nanpercentile(mov[0:10], q_min) if q_min > 0 else np.nanmin(mov)
    index = 0
    #F0 = np.reshape(estimate.b0, dims, order='F')

    components = range(estimate.A.shape[1])
    if discard_bad_components:
        components = estimate.idx_components

    A = estimate.A[:,components]
    cell_contours = video.create_contours(A, dims, magnification, bpx=bpx, thr=thr)
    for frame in mov:
        frame_index = frames[index]
        min_denoised_val = 30
        denoised_gain = 12
        gain_res = 12
        denoised_frame = (np.reshape(A * estimate.C[components, frame_index], dims, order='F')) * denoised_gain + min_denoised_val
        denoised_frame = np.clip(denoised_frame, 0, 255)
        raw_frame = np.clip((frame - minmov) * 255. * gain / (maxmov - minmov), 0, 255)
        ssub = images.shape[1] / B.shape[0]
        B_frame = cv2.resize(B[:, :, index], None,
                             fx=ssub, fy=ssub, interpolation=cv2.INTER_LINEAR)
        contours_frame = np.clip(denoised_gain * (frame - B_frame * 1.0), 0, 255)
        if magnification != 1:
            raw_frame = cv2.resize(raw_frame, None, fx=magnification, fy=magnification,
                                   interpolation=cv2.INTER_LINEAR)
            contours_frame = cv2.resize(contours_frame, None, fx=magnification, fy=magnification,
                                        interpolation=cv2.INTER_LINEAR)
            denoised_frame = cv2.resize(denoised_frame, None, fx=magnification, fy=magnification,
                                        interpolation=cv2.INTER_LINEAR)
        res_frame = cv2.resize(Y_res[:,:,index], dsize=raw_frame.shape[::-1], interpolation=cv2.INTER_LINEAR)

        # Assume residuals should be normaly distributed and show only > 2 std
        res_frame_thr = np.where(res_frame > 2 * np.std(res_frame), res_frame, 0)
        res_frame_thr = np.clip((res_frame_thr - np.min(res_frame_thr)) * gain_res + min_denoised_val, 0, 255)
        denoised_frame = np.reshape(denoised_frame, denoised_frame.shape + (-1,))
        denoised_frame = np.concatenate([denoised_frame,
                                         denoised_frame,
                                         np.zeros_like(denoised_frame)],
                                        axis=2)

        denoised_frame[:, :, 2] = res_frame_thr
        rgbframe = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2RGB)

        contours_frame = cv2.cvtColor(contours_frame.astype('uint8'), cv2.COLOR_GRAY2RGB)
        video.draw_contours(contours_frame, cell_contours, cnm_obj,
                            color_bad_components=(not discard_bad_components))

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
Y_res, B = model_residual(images, cnm_obj, 2, frames, discard_bad_components=True)

avifilename = 'Session' + str(trial_no) + '_' + vid_prefix + str(vid_index) + '_result.avi'
save_movie(cnm_obj.estimates, images, Y_res, B, frames, q_max=75, magnification=2,
           bpx=0, thr=0.6, gain=0.4,
           movie_name=os.path.join(caiman_result_dir, avifilename),
           discard_bad_components=False)
