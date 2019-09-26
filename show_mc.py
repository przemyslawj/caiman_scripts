from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf import params as params
import caiman as cm

import cv2
import numpy as np
import os
import yaml

import video
from miniscope_file import gdrive_download_file, load_session_info

# Choose video
exp_month = '2019-08'
exp_title = 'habituation'
exp_date = '2019-08-27'
animal = 'F-TL'
rootdir = '/home/przemek/neurodata/'
gdrive_subdir = 'cheeseboard-down/down_2'


vid_index = 1
session_index = 1

rclone_config = os.environ['RCLONE_CONFIG']

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

# Download mmap file
mmap_fname = 'msCam' + str(vid_index) + '_els__d1_182_d2_190_d3_1_order_F_frames_1000_.mmap'
local_mmap_fpath = os.path.join(local_session_dir, mmap_fname)
if not os.path.isfile(local_mmap_fpath):
    gdrive_mmap_fpath = os.path.join(gdrive_session_dir, mmap_fname)
    gdrive_download_file(gdrive_mmap_fpath, local_session_dir, rclone_config)

images = video.load_images(local_mmap_fpath)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_movie_fpath = os.path.join(result_dir, 'mc_Session' + str(session_index) + vid_fname)
fr = 3 * 20
dim = images.shape[1:]
out_vid_shape = list(dim[::-1])
out_vid_shape[0] *= 2
out = cv2.VideoWriter(out_movie_fpath, fourcc, fr, tuple(out_vid_shape))
cap = cv2.VideoCapture(local_vid_fpath)

frame_idx = 0
while cap.isOpened():
    ret, org_frame = cap.read()
    if ret:
        mc_frame = cv2.cvtColor(images[frame_idx].astype('uint8'), cv2.COLOR_GRAY2RGB)
        marker_pos = [(50,100), (100, 100)]
        for pos in marker_pos:
            cv2.drawMarker(mc_frame, pos, (255,0,0), markerType=cv2.MARKER_CROSS)
        concat_frame = np.concatenate([org_frame, mc_frame], axis=1)
        out.write(concat_frame)
    else:
        break
    frame_idx += 1

cap.release()
