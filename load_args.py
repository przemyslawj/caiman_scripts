import os


def optional_arg(arg, none_val):
    if arg in os.environ:
        return os.environ[arg]
    return none_val


experiment_month = optional_arg('EXP_MONTH', 'missing')
experiment_title = optional_arg('EXP_TITLE', 'missing')
experiment_date = optional_arg('EXP_DATE', 'missing')
animal_name = optional_arg('ANIMAL', 'missing')
spatial_downsampling = int(optional_arg('DOWNSAMPLE', 2))
downsample_subpath = os.environ['DOWNSAMPLE_SUBPATH']
local_rootdir = os.environ['LOCAL_ROOTDIR']

ncores = int(optional_arg('NCORES', 4))
rclone_config = optional_arg('RCLONE_CONFIG', '')
local_miniscope_path = '/'.join([
    local_rootdir,
    downsample_subpath,
    experiment_month,
    experiment_title,
    experiment_date])
result_data_dir = '/'.join([local_miniscope_path, 'caiman', animal_name])

doPwRigid = True
