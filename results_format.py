import numpy as np
from scipy.io import savemat


def find_centroids(SFP):
    centroids = []
    for cell in range(SFP.shape[2]):
        footprint = SFP[:,:,cell]
        max_val = np.max(footprint)
        x, y = np.where(footprint > max_val / 3)
        centroids.append((int(np.median(x)), int(np.median(y))))
    return centroids


def readSFP(cnm):
    SFP = cnm.estimates.A
    SFP_dims = list(cnm.dims)
    SFP_dims.append(SFP.shape[1])
    SFP = np.reshape(SFP.toarray(), SFP_dims, order='F')
    return SFP


def concat_session_timestamps(session_info):
    mstime = np.array([], dtype=np.int)
    i = 0
    for dat_file in session_info['dat_files']:
        with open(dat_file) as f:
            camNum, frameNum, sysClock, buffer = np.loadtxt(f, dtype='float', comments='#', skiprows=1, unpack=True)
        camNumber = camNum[0]
        mstime_idx = np.where(camNum == camNumber)
        this_mstime = sysClock[mstime_idx]
        this_mstime = this_mstime[0:session_info['session_lengths'][i]]
        mstime = np.concatenate([mstime, this_mstime])
        i += 1

    mstime[0] = 0
    return mstime, camNumber


def save_matlab(cnm, session_info, target_dir, images=[]):
    RawTraces = cnm.estimates.C
    mstime, camNumber = concat_session_timestamps(session_info)
    SFP = readSFP(cnm)
    meanFrame = np.mean(images, axis=0)
    results_dict = {
        # 'dirName': path_to_analyze,
        'Experiment': session_info['animal_name'],
        'numFiles': 1,
        'framesNum': len(RawTraces[1]),
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

    SFPperm = np.transpose(SFP, [2, 0, 1])
    savemat(target_dir + '/SFP.mat', {'SFP': SFPperm})
    savemat(target_dir + '/ms.mat', {'ms': results_dict})