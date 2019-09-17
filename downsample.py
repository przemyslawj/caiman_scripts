import miniscope_file

import os
from moviepy.editor import VideoFileClip


experiment_month = os.environ['EXP_MONTH']
experiment_title = os.environ['EXP_TITLE']
experiment_date = os.environ['EXP_DATE']
animal_name = os.environ['ANIMAL']
spatial_downsampling = int(os.environ['DOWNSAMPLE'])
downsample_subpath = os.environ['DOWNSAMPLE_SUBPATH']
local_rootdir = os.environ['LOCAL_ROOTDIR']

down_size = [752 / spatial_downsampling, 480 / spatial_downsampling]

local_miniscope_path = '/'.join([
    local_rootdir,
    downsample_subpath,
    experiment_month,
    experiment_title,
    experiment_date])

session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, animal_name)
for s_fpath in session_fpaths:
    vids_fpath = miniscope_file.list_vidfiles(s_fpath)
    for video in vids_fpath:
        clip = VideoFileClip(video)
        if clip.size != down_size:
            resized_clip = clip.resize(height=down_size[1], width=down_size[0])
            os.remove(video)
            resized_clip.write_videofile(video, codec='rawvideo')
        else:
            print('Skipping file: ' + video)

