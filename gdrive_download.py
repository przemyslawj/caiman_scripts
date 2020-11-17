import logging

from load_args import *
import miniscope_file

logging.basicConfig(level=logging.INFO)

miniscope_file.gdrive_download_file(os.path.join(src_path, 'rois.csv'), local_miniscope_path, rclone_config)
miniscope_file.gdrive_download_file(os.path.join(src_path, 'cnmfe_params.csv'), local_miniscope_path, rclone_config)
for exp_name in pipeline_setup['experimentNames']:
    miniscope_file.gdrive_download_file(os.path.join(src_path, experiment_title, experiment_date, exp_name, animal_name),
                                        os.path.join(local_miniscope_path, experiment_title, experiment_date, exp_name, animal_name),
                                        rclone_config)
