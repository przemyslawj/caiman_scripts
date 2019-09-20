import miniscope_file

from moviepy.editor import VideoFileClip
from load_args import *

print('local miniscope path: ' + local_miniscope_path)
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

