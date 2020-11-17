import json
import os


pipeline_setup = {}
with open('pipeline_setup.json') as json_file:
    pipeline_setup = json.load(json_file)


def get_config_param(arg, default_val=None):
    if arg in pipeline_setup:
        return pipeline_setup[arg]
    if arg in os.environ:
        return os.environ[arg]
    if default_val is None:
        raise ValueError('No value configured for param=' + arg)
    return default_val


vid_prefix = get_config_param('vidPrefix', 'msCam')

## Run related needs to be in the environ
experiment_title = get_config_param('EXP_TITLE', 'missing_exp_title')
experiment_date = get_config_param('EXP_DATE', 'missing_exp_date')
animal_name = get_config_param('ANIMAL', 'missing')

spatial_downsampling = get_config_param('downsample')
local_rootdir = get_config_param('localTempDirectory')
downsample_subpath = get_config_param('downsampleSubpath')
upload_path = os.path.join(local_rootdir, downsample_subpath)

ncores = get_config_param('ncores', 2)
rclone_config = get_config_param('RCLONE_CONFIG', 'missing_rclone_config')

local_miniscope_path = '/'.join([
    local_rootdir,
    downsample_subpath,
    experiment_title,
    experiment_date])
caiman_result_dir = os.path.join(local_miniscope_path, 'caiman', animal_name)


doPwRigid = True
miniscope_v4 = True

