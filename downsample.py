import logging
import cv2
import numpy as np
import pandas as pd

import miniscope_file
from load_args import *

logging.basicConfig(level=logging.INFO)
replaceVideo = True

logging.info('local miniscope path: ' + local_miniscope_path)

local_rois_fpath = '/'.join([
    local_rootdir,
    downsample_subpath,
    experiment_month,
    'rois.csv'])
if not os.path.isfile(local_rois_fpath):
    raise FileNotFoundError('Roi definition file not found at ' + local_rois_fpath)
rois = pd.read_csv(local_rois_fpath)
animal_roi = rois[rois['animal'] == animal_name]
if len(animal_roi) == 0:
    raise Exception('Roi definition not found for animal: ' + animal_name)
x1 = animal_roi['x1'].values[0]
x2 = animal_roi['x2'].values[0]
width = x2 - x1
y1 = animal_roi['y1'].values[0]
y2 = animal_roi['y2'].values[0]
height = y2 - y1
down_cols = int(width / spatial_downsampling)
down_rows = int(height / spatial_downsampling)

session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, animal_name)
for s_fpath in session_fpaths:
    vids_fpath = miniscope_file.list_vidfiles(s_fpath, vid_prefix)
    for video in vids_fpath:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise IOError('Failed to open input video file ' + video)
        ret, frame = cap.read()
        if not ret:
            raise IOError('Failed to read first frame of input video file ' + video)
        rows, cols = frame.shape[:2]
        logging.debug('Input video shape: (%d, %d)', rows, cols)
        output_vid_path = video[:-4] + '_down.avi'
        if (cols, rows) != (down_cols, down_rows):
            logging.info('Downsampling and cropping file: ' + video)
            fourcc = cv2.VideoWriter_fourcc(*'GREY')
            vid_writer = cv2.VideoWriter(output_vid_path,
                                         fourcc, 60, (down_cols, down_rows),
                                         isColor=False)
            logging.info('Cropped and resized output video shape: (%d, %d)', down_cols, down_rows)
            while ret:
                cropped = frame[y1:y2, x1:x2, 1]
                logging.debug('Cropped shape: %s', str(cropped.shape))
                if spatial_downsampling > 1:
                    resized = cv2.resize(cropped, (down_cols, down_rows))
                else:
                    resized = cropped
                resized = np.uint8(resized)
                logging.debug('Resized shape: %s', str(resized.shape))
                vid_writer.write(resized)
                ret, frame = cap.read()

            vid_writer.release()

            if replaceVideo:
                os.remove(video)
                os.rename(output_vid_path, video)

        else:
            logging.info('Skipping file: ' + video)
        cap.release()

