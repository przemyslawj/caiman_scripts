import miniscope_file
import pandas as pd

from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import crop
from load_args import *

replaceVideo = True

print('local miniscope path: ' + local_miniscope_path)

local_rois_fpath = '/'.join([
    local_rootdir,
    downsample_subpath,
    experiment_month,
    'rois.csv'])
if not os.path.isfile(local_rois_fpath):
    raise FileNotFoundError('Roi definition file not found at ' + local_rois_fpath)
rois = pd.read_csv(local_rois_fpath)
animal_roi = rois[rois['animal'] == animal_name]
if len(animal_roi) == 0:
    raise Exception('Roi definition not found for animal: ' + animal_name)
x1 = animal_roi['x1'].values[0]
x2 = animal_roi['x2'].values[0]
width = x2 - x1
y1 = animal_roi['y1'].values[0]
y2 = animal_roi['y2'].values[0]
height = y2 - y1
down_size = [width / spatial_downsampling, height / spatial_downsampling]

session_fpaths = miniscope_file.list_session_dirs(local_miniscope_path, animal_name)
for s_fpath in session_fpaths:
    vids_fpath = miniscope_file.list_vidfiles(s_fpath, vid_prefix='msCam')
    for video in vids_fpath:
        clip = VideoFileClip(video)
        cropped = crop(clip, x1, y1, x2, y2)
        if cropped.size != down_size:
            resized_clip = cropped.resize(height=down_size[1], width=down_size[0])
            if replaceVideo:
                os.remove(video)
            else:
                video = video[:-4] + '_down.avi'
            resized_clip.write_videofile(video, codec='rawvideo')
        else:
            print('Skipping file: ' + video)
