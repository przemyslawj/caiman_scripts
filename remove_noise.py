#!/usr/bin/env python
# coding: utf-8

# Replaces original video with denoised video.
# Removes two noise issues: horizontal scanning lines and ~3Hz fluctuation in
# brighthness of the entire FOV.
#
# Script adapted from D Aharoni: https://github.com/Aharoni-Lab/Miniscope-v4/blob/master/Miniscope-v4-Denoising-Notebook/V4_Miniscope_noise_removal.ipynb
#

from load_args import *
import miniscope_file

import os.path
from os import path
import cv2
import logging
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, lfilter, freqz, filtfilt


logging.basicConfig(level=logging.INFO)

# TODO: Grab frames per file from metadata
framesPerFile = 1000

# Creates FFT circle mask around center
def create_FFT_Mask(vid_files):
    vid_fpath = vid_files[0]
    cap = cv2.VideoCapture(vid_fpath)
    if not cap.isOpened():
        raise IOError('Failed to open video file ' + vid_fpath)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        raise IOError('Failed to read video ' + vid_fpath)
    rows, cols = frame.shape[:2]

    # Values users can modify:
    goodRadius = 2000
    notchHalfWidth = 3
    centerHalfHeightToLeave = 20

    crow,ccol = int(rows/2) , int(cols/2)

    maskFFT = np.zeros((rows,cols,2), np.float32)
    cv2.circle(maskFFT,(crow,ccol),goodRadius,1,thickness=-1)

    maskFFT[(crow+centerHalfHeightToLeave):,(ccol-notchHalfWidth):(ccol+notchHalfWidth),0] = 0
    maskFFT[:(crow-centerHalfHeightToLeave),(ccol-notchHalfWidth):(ccol+notchHalfWidth),0] = 0

    maskFFT[:,:,1] = maskFFT[:,:,0]
    return maskFFT


# * First run through the data and calculate the mean intensity of every frame.
# * Next apply a lowpass filter to the mean intensity over time
# * Scale the imaging data by the percentage difference of raw and filtered mean intensity.


def spatial_FFT(frame, maskFFT):
    dft = cv2.dft(np.float32(frame),flags = cv2.DFT_COMPLEX_OUTPUT|cv2.DFT_SCALE)
    dft_shift = np.fft.fftshift(dft)

    fshift = dft_shift * maskFFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back


# Calculates mean fluorescence per frame
def calc_mean_frame_vals(vid_fpaths, maskFFT):
    fileNum = 0
    meanFrameList = []
    for vid_path in vid_fpaths:
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            raise IOError('Failed to open video file ' + vid_fpath)
        fileNum += 1
        for frameNum in tqdm(range(0,framesPerFile, 1),
                             total = framesPerFile,
                             desc ="Running file {:.0f}.avi".format(fileNum - 1)):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame[:,:,1]
            #img_back = spatial_FFT(frame, maskFFT)
            #meanFrameList.append(img_back.mean())
            meanFrameList.append(frame.mean())

    meanFrame = np.array(meanFrameList)
    return meanFrame

# Lowpass filter
# Sample rate and desired cutoff frequencies (in Hz).
def lowpass_filter_vals(vals, fs=20, cutoff=3.0, butterOrder=6):
    b, a = butter(butterOrder, cutoff/ (0.5 * fs), btype='low', analog = False)
    return filtfilt(b, a, vals)

# Apply both the 2D FFT spatial filtering and the lowpass mean intensity filtering to the raw data.
def denoise_vids(vid_fpaths, maskFFT, meanFrameFiltered, replace_vid=False):
    vid_format = "GREY"
    codec = cv2.VideoWriter_fourcc(*vid_format)

    fileNum = 0
    for vid_fpath in vid_fpaths:
        fileNum += 1
        data_dir = os.path.dirname(vid_fpath)
        denoised_dir = os.path.join(data_dir, 'denoised')
        miniscope_file.mkdir(denoised_dir)
        cap = cv2.VideoCapture(vid_fpath)
        if not cap.isOpened():
            raise IOError('Failed to open video file ' + vid_fpath)
        filename = os.path.basename(vid_fpath)
        vid_writer = None

        for frameNum in tqdm(range(0, framesPerFile, 1),
                            total = framesPerFile,
                            desc ="Running file {:.0f}.avi".format(fileNum - 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
            ret, frame = cap.read()

            if not ret:
                break

            frame = frame[:,:,1]
            if vid_writer is None:
                rows, cols = frame.shape[:2]
                vid_writer = cv2.VideoWriter(os.path.join(denoised_dir, filename),
                                             codec, 60, (cols, rows), isColor=False)
                logging.debug('Output video shape: %s', str(vid_writer))

            img_back = spatial_FFT(frame, maskFFT)
            meanF = img_back.mean()
            img_back = img_back * (1 + (meanFrameFiltered[frameNum] - meanF)/meanF)
            img_back[img_back > 255] = 255
            img_back = np.uint8(img_back)
            logging.debug('Writing shape: %s', str(img_back.shape))
            logging.debug('Writing min: %s', np.min(img_back))
            logging.debug('Writing max: %s', np.max(img_back))

            vid_writer.write(img_back)

        vid_writer.release()
        if replace_vid:
            os.remove(vid_fpath)
            os.rename(os.path.join(denoised_dir, filename), vid_fpath)

    if replace_vid:
        os.rmdir(denoised_dir)


if __name__ == '__main__':
    session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, animal_name)
    for session_fpath in session_fpaths:
        logging.info('Removing noise in session dir %s', session_fpath)

        vid_fpaths = miniscope_file.list_vidfiles(session_fpath, vid_prefix)
        maskFFT = create_FFT_Mask(vid_fpaths)
        meanFrame = calc_mean_frame_vals(vid_fpaths, maskFFT)
        meanFrameFiltered = lowpass_filter_vals(meanFrame)
        denoise_vids(vid_fpaths, maskFFT, meanFrameFiltered, replace_vid=True)

