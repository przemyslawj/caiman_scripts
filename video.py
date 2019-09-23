import caiman as cm

import cv2
import numpy as np
import os
import skvideo.io


def load_images(memmap_fpath):
    # load memory mappable file
    Yr, dims, T = cm.load_memmap(memmap_fpath)
    images = Yr.T.reshape((T,) + dims, order='F')
    return images


def write_avi(memmap_fpath, result_data_dir):
    images = load_images(memmap_fpath)
    # Write motion corrected video to drive
    w = cm.movie(images)
    #mcwriter = skvideo.io.FFmpegWriter(result_data_dir + '/mc.avi', outputdict={
    #  '-c:v': 'copy'})
    mcwriter = skvideo.io.FFmpegWriter(result_data_dir + '/mc.avi')
    for iddxx, frame in enumerate(w):
      mcwriter.writeFrame(frame.astype('uint8'))
    mcwriter.close()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def mean_frame_avi(f):
    cap = cv2.VideoCapture(f)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mean_frame = np.zeros((height, width), dtype='float64')
    frame_index = 0
    while frame_index < length:
        ret, frame = cap.read()

        if not ret:
            break
        gray_frame = rgb2gray(frame)
        mean_frame += gray_frame / length
    cap.release()
    return mean_frame

if __name__ == '__main__':
    f1 = '/home/przemek/neurodata/cheeseboard-down/down_2/2019-08/habituation/2019-08-27/homecage/mv_caimg/E-BL/Session1/H13_M43_S35/msCam1.avi'
    mean_frame_avi(f1)
