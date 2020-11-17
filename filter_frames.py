#!/usr/bin/env python
# coding: utf-8

import itertools
import bisect
import os.path
import cv2
import logging
import numpy as np
import pandas as pd
import ruptures as rpt
import sys

from matplotlib import pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)

framesPerFile = 1000

# Find points of change for each session's mean frame values
# Returns a list with list of change points per session
def find_change_points(session_vals, penalty=1000):
    vals = np.array(list(itertools.chain(*session_vals)))
    session_lens = [len(x) for x in session_vals]
    total_frames = np.concatenate(([0], np.cumsum(session_lens)))
    algo = rpt.Pelt(model='l2').fit(vals)
    change_points = algo.predict(pen=penalty)
    logging.debug('Changepoints detected: %s', change_points)
    logging.debug('Total frames per session: %s', total_frames)
    session_change_points = [[] for _ in range(len(session_lens))]
    if hasattr(sys, 'ps1'):  # interactive mode
        rpt.display(vals, total_frames, change_points)
        plt.title('Detected change points')
        plt.show(block=True)

    for i, change in enumerate(change_points[:-1]):
        session_i = bisect.bisect_right(total_frames, change) - 1
        session_start_frame = total_frames[session_i]
        session_change_points[session_i].append(change - session_start_frame)

    for i, session_len in enumerate(session_lens):
        session_change_points[i].append(session_len)
    return session_change_points


def get_fragment_means(meanFrame, signal_breaks):
    fragments = list(zip([0] + signal_breaks[:(len(signal_breaks) - 1)], signal_breaks))
    fragmentMeans = [meanFrame[s:t].mean() if t > s else 0 for s, t in fragments]
    return fragments, fragmentMeans


def write_vids(vid_fpaths, mean_frame, valid_frames,
               output_dir='adjusted',
               adjust_gain=True,
               replace_vid=False,
               min_fluorescence_thr=12):
    vid_format = "GREY"
    codec = cv2.VideoWriter_fourcc(*vid_format)
    saved_frames = []

    file_num = 0
    frame_id = 0
    valid_frames_counter = 0
    for vid_fpath in vid_fpaths:
        file_num += 1
        data_dir = os.path.dirname(vid_fpath)
        if os.path.isabs(output_dir):
            denoised_dir = output_dir
        else:
            denoised_dir = os.path.join(data_dir, output_dir)
        if not os.path.isdir(denoised_dir):
            os.mkdir(denoised_dir)
        cap = cv2.VideoCapture(vid_fpath)
        if not cap.isOpened():
            raise IOError('Failed to open video file ' + vid_fpath)
        filename = os.path.basename(vid_fpath)
        vid_writer = None

        for frameNum in tqdm(range(framesPerFile), total=framesPerFile, desc="Running file {:.0f}.avi".format(file_num - 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
            ret, frame = cap.read()

            if not ret:
                break

            img_back = frame[:, :, 1]
            if adjust_gain:
                meanF = img_back.mean()
                img_back = img_back * (1 + (mean_frame - meanF)/meanF)
                img_back[img_back > 255] = 255
            img_back = np.uint8(img_back)

            if vid_writer is None:
                rows, cols = img_back.shape[:2]
                vid_writer = cv2.VideoWriter(os.path.join(denoised_dir, filename),
                                             codec, 60, (cols, rows), isColor=False)
                logging.debug('Output video shape: %s', str(vid_writer))

            if valid_frames_counter < len(valid_frames) and valid_frames[valid_frames_counter] == frame_id:
                valid_frames_counter += 1
                if img_back.mean() >= min_fluorescence_thr:
                    vid_writer.write(img_back)
                    saved_frames.append(frame_id)

            frame_id += 1

        vid_writer.release()
        if replace_vid:
            os.remove(vid_fpath)
            os.rename(os.path.join(denoised_dir, filename), vid_fpath)

    abs_output_dir = denoised_dir
    if replace_vid:
        os.rmdir(denoised_dir)
        abs_output_dir = data_dir
    return abs_output_dir, saved_frames


def is_valid_fragment(fragment_mean, mean_frame, accepted_fluorescence_ratio=0.3):
    return (fragment_mean >= mean_frame * accepted_fluorescence_ratio) and (
            fragment_mean < mean_frame * 1/accepted_fluorescence_ratio)


min_fluorescence_thr=12

if __name__ == '__main__':
    import miniscope_file
    import remove_noise
    from load_args import *
    session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, experiment_date, animal_name)
    session_mean_frame = []
    for session_fpath in session_fpaths:
        logging.info('Calculating mean frame values in session dir %s', session_fpath)
        vid_fpaths = miniscope_file.list_vidfiles(session_fpath, vid_prefix)
        mean_frame = remove_noise.calc_mean_frame_vals(vid_fpaths)
        session_mean_frame.append(mean_frame)

    change_points = find_change_points(session_mean_frame)

    # Calculate mean frame for fragment with max value
    mean_fragment_frame = [0] * len(change_points)
    for session_i in range(len(change_points)):
        fragments, fragmentMeans = get_fragment_means(session_mean_frame[session_i], change_points[session_i])
        validFragments = [i for i, (x, y) in enumerate(fragments) if
                          (fragmentMeans[i] >= min_fluorescence_thr) and (fragments[i][1] - fragments[i][0] > 0)]
        mean_vals = [fragmentMeans[i] for i in validFragments]
        fragment_lens = [fragments[i][1] - fragments[i][0] for i in validFragments]
        if sum(fragment_lens) > 0:
            mean_fragment_frame[session_i] = np.average(mean_vals, weights=fragment_lens)
        else:
            mean_fragment_frame[session_i] = np.nan
            logging.warning('Session has no frames after filtering:' + session_fpaths[session_i])
    mean_frame = np.nanmean(mean_fragment_frame)
    logging.info('Session max mean frame = %f.2', mean_frame)

    for session_i, session_fpath in enumerate(session_fpaths):
        vid_fpaths = miniscope_file.list_vidfiles(session_fpath, vid_prefix)
        fragments, fragmentMeans = get_fragment_means(session_mean_frame[session_i], change_points[session_i])
        validFragments = [i for i, (x, y) in enumerate(fragments) if
                          is_valid_fragment(fragmentMeans[i], mean_frame, 0.3)]
        validFrames = [x for i in validFragments for x in range(fragments[i][0], fragments[i][1])]
        abs_output_dir, saved_frames = write_vids(vid_fpaths, mean_frame, validFrames,
                                                  min_fluorescence_thr=min_fluorescence_thr,
                                                  adjust_gain=False,
                                                  replace_vid=True)

        dat = pd.read_csv(miniscope_file.get_timestamps_fpath(session_fpath))
        valid_dat = dat.iloc[saved_frames]
        valid_dat.to_csv(os.path.join(abs_output_dir, 'timeStamps.csv'))
        nframes_filtered = dat.shape[0] - len(saved_frames)
        if nframes_filtered > 0:
            logging.info('Filtered %d frames from session %s', nframes_filtered, session_fpath)
