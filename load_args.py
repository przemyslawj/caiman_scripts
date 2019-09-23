import os

experiment_month = os.environ['EXP_MONTH']
experiment_title = os.environ['EXP_TITLE']
experiment_date = os.environ['EXP_DATE']
animal_name = os.environ['ANIMAL']
spatial_downsampling = int(os.environ['DOWNSAMPLE'])
downsample_subpath = os.environ['DOWNSAMPLE_SUBPATH']
local_rootdir = os.environ['LOCAL_ROOTDIR']
ncores = int(os.environ['NCORES']) if 'NCORES' in os.environ else 4

local_miniscope_path = '/'.join([
    local_rootdir,
    downsample_subpath,
    experiment_month,
    experiment_title,
    experiment_date])
result_data_dir = '/'.join([local_miniscope_path, 'caiman', animal_name])

doPwRigid = True
