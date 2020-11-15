import logging
import cv2
import numpy as np
import os
import pandas as pd


logging.basicConfig(level=logging.INFO)


def crop_and_downsample(vid_path, crop_roi_xy, rotation_angle, spatial_downsampling,
                        output_vid_dir='down', replace_video=True):
    x1,x2,y1,y2 = crop_roi_xy
    width = x2 - x1
    height = y2 - y1
    down_cols = int(width / spatial_downsampling)
    down_rows = int(height / spatial_downsampling)

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise IOError('Failed to open input video file ' + vid_path)
    ret, frame = cap.read()
    if not ret:
        raise IOError('Failed to read first frame of input video file ' + vid_path)
    rows, cols = frame.shape[:2]
    logging.debug('Input video shape: (%d, %d)', rows, cols)

    if not os.path.isabs(output_vid_dir):
        output_vid_dir = os.path.join(os.path.dirname(vid_path), output_vid_dir)
    if not os.path.isdir(output_vid_dir):
        os.mkdir(output_vid_dir)
    output_vid_path = os.path.join(output_vid_dir,
                                   os.path.basename(vid_path))
    if (cols, rows) == (down_cols, down_rows):
        cap.release()
        return None

    logging.info('Downsampling and cropping file: ' + vid_path)
    fourcc = cv2.VideoWriter_fourcc(*'GREY')
    vid_writer = cv2.VideoWriter(output_vid_path,
                                 fourcc, 60, (down_cols, down_rows),
                                 isColor=False)
    logging.info('Cropped and resized output video shape: (%d, %d)', down_cols, down_rows)

    rotation_matrix = cv2.getRotationMatrix2D((rows/2, cols/2), rotation_angle, 1)
    while ret:
        if rotation_angle > 0:
            frame = cv2.warpAffine(frame, rotation_matrix, (cols, rows))
            logging.debug('Rotated video to shape: (%d, %d)', frame.shape[:2])
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
    cap.release()

    if replace_video:
        os.remove(vid_path)
        os.rename(output_vid_path, vid_path)
        output_vid_path = vid_path
    return output_vid_path


def get_roi_xy(local_rois_fpath, animal_name):
    if not os.path.isfile(local_rois_fpath):
        raise FileNotFoundError('Roi definition file not found at ' + local_rois_fpath)
    rois = pd.read_csv(local_rois_fpath)
    animal_roi = rois[rois['animal'] == animal_name]
    if len(animal_roi) == 0:
        raise Exception('Roi definition not found for animal: ' + animal_name)
    x1 = animal_roi['x1'].values[0]
    x2 = animal_roi['x2'].values[0]
    y1 = animal_roi['y1'].values[0]
    y2 = animal_roi['y2'].values[0]
    rotation = 0
    if 'rotation' in animal_roi.columns:
        rotation = animal_roi['rotation'].values[0]
    return (x1, x2, y1, y2), rotation

if __name__ == '__main__':
    from load_args import *
    import miniscope_file
    local_rois_fpath = '/'.join([
        local_rootdir,
        downsample_subpath,
        experiment_month,
        'rois.csv'])
    logging.info('local miniscope path: ' + local_miniscope_path)

    crop_roi, rotation = get_roi_xy(local_rois_fpath, animal_name)

    session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, animal_name)
    for s_fpath in session_fpaths:
        vids_fpath = miniscope_file.list_vidfiles(s_fpath, vid_prefix)
        for video in vids_fpath:
            output_vid_path = crop_and_downsample(
                    video, crop_roi, rotation, spatial_downsampling)
            if output_vid_path is None:
                logging.info('Video file unchanged: ' + video)

