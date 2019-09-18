import caiman as cm

import os
import skvideo.io

experiment_month = os.environ['EXP_MONTH']
experiment_title = os.environ['EXP_TITLE']
experiment_date = os.environ['EXP_DATE']
animal_name = os.environ['ANIMAL']
spatial_downsampling = int(os.environ['DOWNSAMPLE'])
downsample_subpath = os.environ['DOWNSAMPLE_SUBPATH']
local_rootdir = os.environ['LOCAL_ROOTDIR']
local_miniscope_path = '/'.join([
    local_rootdir,
    downsample_subpath,
    experiment_month,
    experiment_title,
    experiment_date])
result_data_dir = '/'.join([local_miniscope_path, 'caiman', animal_name])


def write_avi(memmap_fpath, result_data_dir):
    # load memory mappable file
    Yr, dims, T = cm.load_memmap(memmap_fpath)
    images = Yr.T.reshape((T,) + dims, order='F')

    # Write motion corrected video to drive
    w = cm.movie(images)
    mcwriter = skvideo.io.FFmpegWriter(result_data_dir + '/mc.avi', outputdict={
      '-c:v': 'copy'})
    #mcwriter = skvideo.io.FFmpegWriter(result_data_dir + '/mc.avi')
    for iddxx, frame in enumerate(w):
      mcwriter.writeFrame(frame.astype('uint8'))
    mcwriter.close()


if __name__ == '__main__':
    import miniscope_file
    memmap_fpath = miniscope_file.get_joined_memmap_fpath(result_data_dir)
    write_avi(memmap_fpath, result_data_dir)
