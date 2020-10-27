import os

def optional_arg(arg, none_val):
    if arg in os.environ:
        return os.environ[arg]
    return none_val


vid_prefix = optional_arg('VID_PREFIX', 'msCam')
experiment_month = optional_arg('EXP_MONTH', 'missing_exp_month')
experiment_title = optional_arg('EXP_TITLE', 'missing_exp_title')
experiment_date = optional_arg('EXP_DATE', 'missing_exp_date')
animal_name = optional_arg('ANIMAL', 'missing')
spatial_downsampling = int(optional_arg('DOWNSAMPLE', 2))
downsample_subpath = os.environ['DOWNSAMPLE_SUBPATH']
local_rootdir = os.environ['LOCAL_ROOTDIR']

ncores = int(optional_arg('NCORES', 4))
rclone_config = optional_arg('RCLONE_CONFIG', 'missing_rclone_config')
local_miniscope_path = '/'.join([
    local_rootdir,
    downsample_subpath,
    experiment_month,
    experiment_title,
    experiment_date])
result_data_dir = '/'.join([local_miniscope_path, 'caiman', animal_name])

doPwRigid = True
miniscope_v4 = True

