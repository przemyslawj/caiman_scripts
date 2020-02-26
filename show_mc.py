import caiman as cm

import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip

import video
from miniscope_file import gdrive_download_file, load_session_info

# Choose video
exp_month = '2020-01'
exp_title = 'habituation'
exp_date = '2020-01-29'
animal = 'G-BR'
#rootdir = '/mnt/DATA/Prez/'
rootdir = '/home/prez/neurodata'
gdrive_subdir = 'cheeseboard-down/down_2'
pwRigid = True

vid_index = 1
session_index = 1

rclone_config = os.environ['RCLONE_CONFIG']

local_rois_fpath = '/'.join([
    rootdir,
    gdrive_subdir,
    exp_month,
    'rois.csv'])

"""Prepare data"""
gdrive_dated_dir = os.path.join(gdrive_subdir, exp_month, exp_title, exp_date)
local_dated_dir = os.path.join(rootdir, gdrive_dated_dir)
gdrive_result_dir = os.path.join(gdrive_dated_dir, 'caiman', animal)
result_dir = os.path.join(local_dated_dir, 'caiman', animal)
session_info = load_session_info(result_dir, gdrive_result_dir, rclone_config)

# Download downsampled video
server_session_dir = os.path.dirname(session_info['dat_files'][session_index - 1])
timestamped_session_subdir = server_session_dir.split(exp_date + '/')[1]
local_session_dir = os.path.join(local_dated_dir, timestamped_session_subdir)
vid_fname = 'msCam' + str(vid_index) + '.avi'
local_vid_fpath = os.path.join(local_session_dir, vid_fname)
gdrive_session_dir = '/'.join([gdrive_subdir, exp_month, exp_title, exp_date, timestamped_session_subdir])
if not os.path.isfile(local_vid_fpath):
    gdrive_vid_fpath = os.path.join(gdrive_session_dir, vid_fname)
    gdrive_download_file(gdrive_vid_fpath, local_session_dir, rclone_config)
orgVideo = VideoFileClip(local_vid_fpath)

# Download mmap file
mmap_prefix = 'rig'
dims = orgVideo.size[::-1]
if pwRigid:
    mmap_prefix = 'els'
mmap_fname = 'msCam' + str(vid_index) + '_' + mmap_prefix + '__d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_1_order_F_frames_1000_.mmap'
local_mmap_fpath = os.path.join(local_session_dir, mmap_fname)
if not os.path.isfile(local_mmap_fpath):
    gdrive_mmap_fpath = os.path.join(gdrive_session_dir, mmap_fname)
    gdrive_download_file(gdrive_mmap_fpath, local_session_dir, rclone_config)


# Show optical flow
def dispOpticalFlow(image, flow, divisor=20):
    """Display image with a visualisation of a flow over the top. A divisor controls the density of the quiver plot."""
    img_shape = np.shape(image)
    # determine number of quiver points there will be
    imax = int(img_shape[0] / divisor)
    jmax = int(img_shape[1] / divisor)
    # create a blank mask, on which lines will be drawn.
    mask = np.zeros_like(image)
    for i in range(1, imax):
        for j in range(1, jmax):
            x1 = (i) * divisor
            y1 = (j) * divisor
            flow_gain = 10
            x2 = int(x1 + flow[x1, y1, 1] * flow_gain)
            y2 = int(y1 + flow[x1, y1, 0] * flow_gain)
            x2 = np.clip(x2, 0, img_shape[0])
            y2 = np.clip(y2, 0, img_shape[1])
            # Add yellow arrows
            mask = cv2.arrowedLine(mask, (y1, x1), (y2, x2), (100, 0, 0), thickness=2)

    # superpose lines onto image
    return mask
    #return cv2.add((image / np.max(image) * 2).astype('uint8'), mask.astype('uint8'))


tmpl_els, correlations_els, flows_els, norms_els, crispness_els = cm.motion_correction.compute_metrics_motion_correction(
    local_mmap_fpath, dims[0], dims[1],
    False, play_flow=False, resize_fact_flow=0.2)

images = video.load_images(local_mmap_fpath)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_movie_fpath = os.path.join(result_dir, 'mc_Session' + str(session_index) + vid_fname)
fr = 3 * 20
dim = images.shape[1:]
out_vid_shape = list(dim[::-1])
out_vid_shape[0] *= 3
out = cv2.VideoWriter(out_movie_fpath, fourcc, fr, tuple(out_vid_shape))
cap = cv2.VideoCapture(local_vid_fpath)


def drawGrid(frame):
    grid_spacing_x = round(frame.shape[0] / 4)
    grid_spacing_y = round(frame.shape[1] / 4)
    for x_pos in range(grid_spacing_x, frame.shape[0] - 1, grid_spacing_x):
        cv2.line(frame, (x_pos, 0), (x_pos, frame.shape[1]), color=(255, 0, 0), thickness=1)
    for y_pos in range(grid_spacing_y, frame.shape[1] - 1, grid_spacing_y):
        cv2.line(frame, (0, y_pos), (frame.shape[0], y_pos), color=(255, 0, 0), thickness=1)


frame_idx = 0
nframes = images.shape[0]
nflow_frames = len(flows_els)
nframes_per_flow = nframes / nflow_frames
flow_frame = None
while cap.isOpened():
    ret, org_frame = cap.read()
    if ret:
        frame = cv2.cvtColor(images[frame_idx].astype('uint8'), cv2.COLOR_GRAY2RGB)
        mc_frame = np.copy(frame)
        drawGrid(mc_frame)

        drawGrid(org_frame)

        if frame_idx % nframes_per_flow == 0:
            flow3d = flows_els[int(frame_idx / nframes_per_flow)]
            flow_frame = np.copy(frame)
            flow_frame = dispOpticalFlow(flow_frame, flow3d)
        concat_frame = np.concatenate([org_frame, mc_frame, flow_frame], axis=1)
        out.write(concat_frame)
    else:
        break
    frame_idx += 1

cap.release()
