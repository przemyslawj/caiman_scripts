import miniscope_file

import os
from moviepy.editor import VideoFileClip


experiment_month = '2019-08'
experiment_title = 'habituation'
experiment_date = '2019-08-27' #@param {type: "string"}
animal_name = 'E-BL'  #@param {type: "string"}
spatial_downsampling = 2 #@param {type: int}
down_size = (752 / spatial_downsampling, 480 / spatial_downsampling)

local_miniscope_path = '/'.join([
    '/mnt/DATA/Prez/cheeseboard-down/down_' + str(spatial_downsampling),
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

