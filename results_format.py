import logging
import numpy as np
import pandas as pd
import os
from scipy.io import savemat
from scipy.ndimage import center_of_mass

from miniscope_file import gdrive_download_file

def find_centroids(SFP):
    return [center_of_mass(SFP[:, :, ii]) for ii in range(SFP.shape[2])]


def readSFP(cnm, filter_components=False):
    SFP = cnm.estimates.A
    if filter_components:
        SFP = SFP[:, cnm.estimates.idx_components]
    SFP_dims = list(cnm.dims)
    SFP_dims.append(SFP.shape[1])
    SFP = np.reshape(SFP.toarray(), SFP_dims, order='F')
    return SFP


def read_dat_timestamps(dat_file):
    with open(dat_file) as f:
        camNum, frameNum, sysClock, buffer_ = np.loadtxt(f, dtype='float', comments='#', skiprows=1, unpack=True)
    camNumber = camNum[0]
    mstime_idx = np.where(camNum == camNumber)
    this_mstime = sysClock[mstime_idx]
    return this_mstime, camNumber


def read_timestamps(timestamp_file):
    if timestamp_file.endswith('.dat'):
        return read_dat_timestamps(timestamp_file)
    dat = pd.read_csv(timestamp_file)
    ts = dat['Time Stamp (ms)'].values
    ts[ts < 0] = 0
    return ts, [0] * dat.shape[0]


def concat_session_timestamps(session_info, rootdir, gdrive_subdir, rclone_config):
    mstime = np.array([], dtype=np.int)
    i = 0
    for timestamp_file in session_info['timestamp_files']:
        #TODO: temporary replacement until session_info.yaml files updated
        timestamp_file = timestamp_file.replace('/test/', '/beforetest/')
        if not os.path.isfile(timestamp_file):
            gdrive_dat_fpath = timestamp_file[timestamp_file.find(gdrive_subdir):]
            timestamp_file = os.path.join(rootdir, gdrive_dat_fpath)
            gdrive_download_file(gdrive_dat_fpath, os.path.dirname(timestamp_file), rclone_config)
        this_mstime, camNumber = read_timestamps(timestamp_file)
        this_mstime = this_mstime[0:session_info['session_lengths'][i]]
        missing_len = len(this_mstime) - session_info['session_lengths'][i]
        if missing_len > 0:
            logging.warn('Too few timestamps recorded in file %s, missing %d timestamps', timestamp_file, missing_len)
            avg_timediff = int(np.mean(np.diff(this_mstime)))
            this_mstime = np.concatenate([this_mstime,
                np.linspace(avg_timediff, avg_timediff * missing_len, avg_timediff) + this_mstime[-1]])

        mstime = np.concatenate([mstime, this_mstime])
        i += 1

    mstime[0] = 0
    return mstime, camNumber


def save_matlab(cnm, session_info, target_dir, images, mstime, camNumber, extraFields={}):
    RawTraces = cnm.estimates.C
    SFP = readSFP(cnm)
    meanFrame = np.mean(images, axis=0)
    results_dict = {
        # 'dirName': path_to_analyze,
        'Experiment': session_info['animal_name'],
        'numFiles': 1,
        'framesNum': len(RawTraces[0]),
        'maxFramesPerFile': 1000,
        'height': cnm.dims[0],
        'width': cnm.dims[1],
        'camNumber': camNumber,
        'time': mstime,
        'sessionLengths': session_info['session_lengths'],
        # 'analysis_time': analysis_time,
        'meanFrame': meanFrame,
        'Centroids': find_centroids(SFP),
        #'CorrProj': cn_filter,
        #'PeakToNoiseProj': pnr,
        'PNR': cnm.estimates.SNR_comp, # Calculated Peak-to-Noise ratios
        'cnn_preds': cnm.estimates.cnn_preds, # CNN prediction probability
        'neurons_sn': cnm.estimates.neurons_sn, # neurons noise estimation
        'RawTraces': RawTraces.conj().transpose(),  # swap time x neurons dimensions
        # 'FiltTraces': cnm.estimates.F_dff,
        'DeconvTraces': cnm.estimates.S.conj().transpose(),
        'SFPs': SFP,
        'numNeurons': SFP.shape[2],
        # 'analysis_duration': analysis_duration
    }
    results_dict.update(extraFields)
    SFPperm = np.transpose(SFP, [2, 0, 1])
    savemat(target_dir + '/SFP.mat', {'SFP': SFPperm})
    savemat(target_dir + '/ms.mat', {'ms': results_dict})

    return target_dir + '/ms.mat', target_dir + '/SFP.mat'

