import numpy as np
from caiman.source_extraction.cnmf.initialization import downscale
from scipy.sparse import csr_matrix


def model_residual(images, cnm, ssub_B, frames=[], discard_bad_components=True):
    dims = images.shape[1:]
    if len(frames) == 0:
        frames = range(images.shape[0])
    T = images.shape[0]

    components = range(cnm.estimates.A.shape[1])
    if discard_bad_components:
        components = cnm.estimates.idx_components
    # Y_res = Y - AC - B, where B = b0 + W(Y-AC-b0)
    AC = cnm.estimates.A[:,components].dot(cnm.estimates.C[components,:][:,frames])
    AC_ds = downscale(AC.reshape(dims + (-1,), order='F'),
                      (ssub_B, ssub_B, 1))

    W = cnm.estimates.W
    if type(W) == dict:
        W = csr_matrix((W['data'], W['indices'], W['indptr']), shape=W['shape'])

    b0_ds = downscale(cnm.estimates.b0.reshape(dims, order='F'),
                      (ssub_B, ssub_B))
    ds_dims = AC_ds.shape
    b0_ds_rep = np.tile(b0_ds, (T, 1, 1)).transpose((1, 2, 0))
    images_ds = downscale(images, (1, ssub_B, ssub_B)).transpose((1, 2, 0))
    B = b0_ds_rep + W.dot((images_ds - AC_ds - b0_ds_rep).reshape((W.shape[0], T), order='F')).reshape(ds_dims, order='F')
    Y_res = images_ds - AC_ds - B

    return Y_res