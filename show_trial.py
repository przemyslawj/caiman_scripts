import cv2
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from miniscope_file import gdrive_download_file, load_session_info, load_hdf5_result
from load_args import *
import video


logging.basicConfig(level=logging.INFO)

trial = int(os.environ['TRIAL_NO'])
selected_cell_ids_str = optional_arg('CELL_IDS', '')

selected_cell_ids = []
if len(selected_cell_ids_str) > 0:
    selected_cell_ids = [int(x) for x in selected_cell_ids_str.split(',')]

useFiltered = True

exp_name = 'beforetest'
dated_dir = os.path.join(local_rootdir, 'cheeseboard', experiment_month, experiment_title, experiment_date, exp_name)


def create_video_writer(video_outputfile, frame_rate, format='IYUV', video_dim=(640,480)):
    if hasattr(cv2, 'VideoWriter_fourcc'):
        fourcc = cv2.VideoWriter_fourcc(*format)
    else:
        fourcc = cv2.CV_FOURCC(*format)

    video_writer = cv2.VideoWriter(video_outputfile, fourcc, frame_rate, video_dim)
    if not video_writer.isOpened():
        raise IOError('Failed to open video writer to: ' + video_outputfile)
    return video_writer


def get_vector(angle_deg, vec_length):
    angle_rad = angle_deg / 180 * np.pi
    y = np.sin(angle_rad) * vec_length
    x = np.cos(angle_rad) * vec_length
    return x, y


def stream_from_file(video_file):
    stream = cv2.VideoCapture()
    video_opened = stream.open(video_file)
    if not video_opened:
        IOError('Couldnt open the video')

    return stream


# Prepare tracking video stream

experiment_date_prefix = experiment_date
if exp_name == 'beforetest':
    experiment_date_prefix = experiment_date + '-beforetest'
video_filename = '_'.join(filter(lambda x: x is not None, [experiment_date_prefix, animal_name, 'trial', str(trial)])) + '.avi'
movie_dir = os.path.join(dated_dir, 'movie')
video_filepath = os.path.join(movie_dir, video_filename)
if not os.path.isfile(video_filepath):
    gdrive_vid_dir = '/'.join(['cheeseboard', experiment_month, experiment_title, experiment_date, exp_name, 'movie'])
    gdrive_vid_fpath = os.path.join(gdrive_vid_dir, video_filename)
    gdrive_download_file(gdrive_vid_fpath, movie_dir, rclone_config)

stream = stream_from_file(video_filepath)
has_frame, frame = stream.read()
if not has_frame:
    raise IOError('No frames could be read from the file stream: ' + video_filepath)

# # Prepare data frame with reward positions
# loc_filepath = os.path.join(dated_dir, 'locations.csv')
# if os.path.exists(loc_filepath):
#     loc_df = pd.read_csv(loc_filepath)

# Prepare ca img video stream
gdrive_dated_dir = os.path.join(upload_path,  experiment_month, experiment_title, experiment_date)
local_dated_dir = os.path.join(local_rootdir, downsample_subpath, experiment_month, experiment_title, experiment_date)
gdrive_result_dir = os.path.join(gdrive_dated_dir, 'caiman', animal_name)
result_dir = os.path.join(local_dated_dir, 'caiman', animal_name)
# Download result files if not stored locally
cnm_obj = load_hdf5_result(result_dir, gdrive_result_dir, rclone_config, use_filtered=useFiltered)

if len(selected_cell_ids) > 0:
    selected_cells = np.where(np.isin(cnm_obj.estimates.registered_cell_ids, selected_cell_ids))[0]
else:
    selected_cells = [x for x in range(10, 100, 5)]

session_info = load_session_info(result_dir, gdrive_result_dir, rclone_config)
session_lengths = np.cumsum([0] + session_info['session_lengths'])
session_trace_offset = session_lengths[trial - 1]

dims = cnm_obj.estimates.dims
mmap_prefix = 'els'
n_mscam_vids = int(np.ceil(session_info['session_lengths'][trial - 1] / 1000))
local_mmap_fpaths = []
for vid_index in range(1, n_mscam_vids + 1):
    frames = 1000
    if vid_index == n_mscam_vids:
        frames = session_info['session_lengths'][trial - 1] % 1000
    mmap_fname = 'msCam' + str(vid_index) + '_' + mmap_prefix + \
                 '__d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_1_order_F_frames_' + \
                 str(frames) + '_.mmap'
    remote_mmap_dir = os.path.dirname(session_info['dat_files'][trial - 1])
    mmap_session_subdir = remote_mmap_dir.split(experiment_date + '/')[1]
    local_mmap_dir = os.path.join(local_dated_dir, mmap_session_subdir)
    local_mmap_fpath = os.path.join(local_mmap_dir, mmap_fname)
    if not os.path.isfile(local_mmap_fpath):
        gdrive_mmap_dir = '/'.join([upload_path,  experiment_month, experiment_title, experiment_date, mmap_session_subdir])
        gdrive_mmap_fpath = os.path.join(gdrive_mmap_dir, mmap_fname)
        if gdrive_download_file(gdrive_mmap_fpath, local_mmap_dir, rclone_config):
            local_mmap_fpaths.append(local_mmap_fpath)
    else:
        local_mmap_fpaths.append(local_mmap_fpath)

local_timestamp_fpath = os.path.join(local_mmap_dir, 'timestamp.dat')
if not os.path.isfile(local_timestamp_fpath):
    gdrive_timestamp_dir = '/'.join(['cheeseboard',  experiment_month, experiment_title, experiment_date, mmap_session_subdir])
    gdrive_timestamp_fpath = os.path.join(gdrive_timestamp_dir, 'timestamp.dat')
    gdrive_download_file(gdrive_timestamp_fpath, local_mmap_dir, rclone_config)
import results_format
caimg_timestamps, _ = results_format.read_timestamps(local_timestamp_fpath)

# Prepare data frame with tracking positions
tracking_filename = '_'.join(filter(lambda x: x is not None, [experiment_date, animal_name, 'trial', str(trial), 'positions.csv']))
tracking_dir = os.path.join(movie_dir, 'tracking')
tracking_filepath = os.path.join(tracking_dir, tracking_filename)
if not os.path.isfile(tracking_filepath):
    gdrive_tracking_dir = '/'.join(['cheeseboard', experiment_month, experiment_title, experiment_date, exp_name, 'movie', 'tracking'])
    gdrive_tracking_fpath = os.path.join(gdrive_tracking_dir, tracking_filename)
    gdrive_download_file(gdrive_tracking_fpath, tracking_dir, rclone_config)
tracking_df = pd.read_csv(tracking_filepath)

video_traces_outputfile = os.path.join(tracking_dir, tracking_filename + '_traces.avi')

video_dims = (320 * 2, 240)
single_video_dims = (320, 240)
caimg_frame_rate = 20
#video_outputfile = os.path.join(tracking_dir, tracking_filename + '.avi')
#video_writer = create_video_writer(video_outputfile, frame_rate=15*3,
#                                   format='FFV1',
#                                   video_dim=video_dims)

frame_idx = 0
row_idx = 0
window_name = '_'.join([experiment_date, animal_name, str(trial)])
valid_pos = [(0, 0)] * len(tracking_df)
pos_idx = 0
caimg_frame = 0

ca_images = np.concatenate([video.load_images(f) for f in local_mmap_fpaths], axis=0)
ca_movie = ca_images
mean_movie = np.mean(ca_movie, axis=0)

maxmov = int(np.nanpercentile(ca_movie[10:50], 80))
minmov = int(np.nanpercentile(ca_movie[10:50], 2))
caimg_frame = None
caimg_timestamps[0] = -1
caimg_frame_index = 0
caimg_timestamp = -1

writer = FFMpegWriter(fps=15 * 3, metadata=dict(title='Test'))
fig = plt.figure()
vid_ax = plt.subplot(2, 1, 1)
ca_vid_plt = plt.imshow(np.zeros(video_dims).transpose())
vid_ax.set_axis_off()

trace_ax = plt.subplot(2, 1, 2)
window_sec = 4
plt.xlim(-window_sec, window_sec)
C = cnm_obj.estimates.C
cell_contours = video.create_contours(cnm_obj.estimates.A[:,selected_cells],
                                      ca_movie.shape[1:], thr=0.75)
trace_vid_plt = [0] * len(selected_cells)
contour_colours = dict()
# length of the plotted trace. Extend the window by 1 sec because the timestamps aren't evenly spread.
trace_len = 2 * (window_sec + 1) * caimg_frame_rate + 1
for i in range(len(selected_cells)):
    cell_label = str(selected_cells[i])
    if useFiltered:
        cell_label = str(cnm_obj.estimates.registered_cell_ids[selected_cells[i]])

    trace_vid_plt[i], = trace_ax.plot(np.arange(trace_len),
                                      np.arange(trace_len),
                                      '-',
                                      label=cell_label)
    hexcol = trace_vid_plt[i].get_c()
    contour_colours[i] = np.array(mcolors.hex2color(hexcol)) * 255

ca_max = np.quantile(C[selected_cells,:], 0.995) + len(selected_cells) * 2.0
ca_min = np.quantile(C[selected_cells,:], 0.05)
plt.ylim(ca_min, ca_max)
trace_ax.legend(loc='upper left', fontsize='x-small')
trace_ax.axes.get_yaxis().set_visible(False)
trace_ax.patch.set_visible(False)

writer.setup(fig, video_traces_outputfile, dpi=500)
print('Output file: ' + video_traces_outputfile)
while has_frame:
    if row_idx >= len(tracking_df):
        break

    if frame_idx == tracking_df.loc[row_idx, 'frame']:
        pos_x = tracking_df.loc[row_idx, 'smooth_x']
        pos_y = tracking_df.loc[row_idx, 'smooth_y']
        if pos_x >= 0 and pos_y >= 0:
            valid_pos[pos_idx] = (int(pos_x), int(pos_y))
            pos_idx += 1
        row_idx += 1

    for i in range(1, pos_idx):
        cv2.line(frame, valid_pos[i-1], valid_pos[i], (255, 0, 0), 2)

    if 'smooth_heading_angle' in tracking_df.columns:
        angle = tracking_df.loc[row_idx - 1, 'smooth_heading_angle']
        if angle != 360:
            arrow_start = valid_pos[pos_idx - 1]
            v = get_vector(angle, 15)
            arrow_end = np.round(np.array(arrow_start) + np.array(v))
            cv2.arrowedLine(frame, arrow_start, (int(arrow_end[0]), int(arrow_end[1])), (0, 255, 255), 3)
    cv2.putText(frame, "Time: " + str(int(tracking_df.loc[row_idx - 1, 'timestamp']/1000)), (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    while caimg_frame_index < ca_movie.shape[0] and \
            caimg_timestamps[caimg_frame_index] < tracking_df.loc[row_idx - 1, 'timestamp']:
        caimg_frame = ca_movie[caimg_frame_index]
        caimg_frame_index += 1

    caimg_frame_clipped = np.clip((caimg_frame - minmov) * 255. * 0.6 / (maxmov - minmov), 0., 255.)
    #caimg_frame_dff = np.clip(0.2 * caimg_frame + (caimg_frame - mean_movie) / mean_movie * 250, 0., 255.)
    rgb_caimg_frame = cv2.cvtColor(caimg_frame_clipped.astype('uint8'), cv2.COLOR_GRAY2RGB)
    video.draw_contours(rgb_caimg_frame, cell_contours, cnm_obj, contour_colours)

    merged_frame = np.concatenate(
        [cv2.resize(frame, single_video_dims), cv2.resize(rgb_caimg_frame, single_video_dims)],
        axis=1)
    #cv2.imshow(window_name, rgb_caimg_frame.astype('uint8'))

    ca_vid_plt.set_array(merged_frame.astype('uint8'))

    def get_centered_timestamps(timestamps, center_index, nframes=window_sec * caimg_frame_rate):
        indices = range(max(0, center_index - nframes), min(center_index + nframes, len(timestamps)))
        ts = (np.array(timestamps[[*indices]]) - timestamps[center_index]) / 1000
        return indices, ts

    indices, ts = get_centered_timestamps(caimg_timestamps, caimg_frame_index - 1)
    for cell_i, cell in enumerate(selected_cells):
        shifted_trace = C[cell, np.array(indices) + session_trace_offset] + cell_i * 2.0
        trace_len_diff = trace_len - len(shifted_trace)
        cell_ts = ts
        if trace_len_diff > 0:
            logging.debug('Expanding trace len by %d', trace_len_diff)
            cell_ts = np.append(ts, [window_sec + np.arange(trace_len_diff)])
            shifted_trace = np.append(shifted_trace, [0] * trace_len_diff)

        trace_vid_plt[cell_i].set_data(cell_ts, shifted_trace)

    if pos_idx > 0:
        #video_writer.write(merged_frame.astype('uint8'))
        writer.grab_frame()

    #k = cv2.waitKey(100) & 0xff
    #if k == 27:  # ESC
    #    break

    #if frame_idx > 500:
    #    break
    has_frame, frame = stream.read()
    frame_idx += 1

writer.finish()
#cv2.destroyAllWindows()
#video_writer.release()
stream.release()
