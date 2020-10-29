import caiman as cm

import cv2
import numpy as np
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
    mcwriter = skvideo.io.FFmpegWriter(result_data_dir + '/mc.avi',
                                       outputdict={'-vcodec': 'rawvideo'})
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


def create_contours(A, frame_dims, magnification=1, bpx=0, thr=0.8):
    cell_contours = dict()
    cell_idx = 0
    for a in A.T.toarray():
        a = a.reshape(frame_dims, order='F')
        if bpx > 0:
            a = a[bpx:-bpx, bpx:-bpx]
        if magnification != 1:
            a = cv2.resize(a, None, fx=magnification, fy=magnification,
                           interpolation=cv2.INTER_LINEAR)
        ret, thresh = cv2.threshold(a, thr * np.max(a), 1., 0)
        img, contour, hierarchy = cv2.findContours(
            thresh.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        contours.append(contour)
        contours.append(list([c + np.array([[a.shape[1], 0]]) for c in contour]))
        contours.append(list([c + np.array([[2 * a.shape[1], 0]]) for c in contour]))
        cell_contours[cell_idx] = contours
        cell_idx += 1
    return cell_contours


def draw_contours(frame, cell_contours, cnm_obj, colours={}, color_bad_components=False):
    for cell_idx in cell_contours.keys():
        yellow_col = (0, 255, 255)
        red_col = (0, 0, 255)
        contour_col = yellow_col
        if cell_idx in colours.keys():
            contour_col = colours[cell_idx]
        if color_bad_components and (cell_idx in cnm_obj.estimates.idx_components_bad):
            contour_col = red_col

        for contour in cell_contours[cell_idx]:
            convexContour = cv2.convexHull(contour[0], clockwise=True)
            cv2.drawContours(frame, [convexContour], -1, contour_col, 1)


if __name__ == '__main__':
    f1 = '/home/przemek/neurodata/cheeseboard-down/down_2/2019-08/habituation/2019-08-27/homecage/mv_caimg/E-BL/Session1/H13_M43_S35/msCam1.avi'
    mean_frame_avi(f1)
