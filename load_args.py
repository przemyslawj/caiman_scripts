import json
import os


pipeline_setup = {}
with open('pipeline_setup.json') as json_file:
    pipeline_setup = json.load(json_file)


def get_config_param(arg, default_val=None):
    if arg in os.environ:
        return os.environ[arg]
    if arg in pipeline_setup:
        return pipeline_setup[arg]
    if default_val is None:
        raise ValueError('No value configured for param=' + arg)
    return default_val


vid_prefix = get_config_param('vidPrefix', 'msCam')
spatial_downsampling = get_config_param('downsample')
ncores = get_config_param('ncores', 2)
rclone_config = get_config_param('rcloneConfig')

## Paths
src_path = get_config_param('sourceDirectory')
local_rootdir = get_config_param('localTempDirectory')
downsample_subpath = get_config_param('downsampleSubpath')
upload_path = get_config_param('uploadDirectory')
local_miniscope_path = os.path.join(local_rootdir, downsample_subpath)
caiman_src_datadir = get_config_param('caimanSrcDatadir')

## Run related needs to be in the environ
experiment_title = get_config_param('EXP_TITLE', 'missing_exp_title')
experiment_date = get_config_param('EXP_DATE', 'missing_exp_date')
animal_name = get_config_param('ANIMAL', 'missing')
caiman_result_dir = os.path.join(local_miniscope_path, experiment_title, experiment_date, 'caiman', animal_name)


doPwRigid = True
is_v4_path = get_config_param('isV4Path', 0) == 1
