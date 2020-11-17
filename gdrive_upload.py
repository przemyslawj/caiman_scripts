import logging

from load_args import *
import miniscope_file

logging.basicConfig(level=logging.INFO)

for exp_name in pipeline_setup['experimentNames']:
    copied_dir = os.path.join(local_miniscope_path, experiment_title, experiment_date, exp_name, animal_name)
    if os.path.isdir(copied_dir):
        miniscope_file.gdrive_upload_file(copied_dir,
                                          os.path.join(upload_path, experiment_title, experiment_date, exp_name, animal_name),
                                          rclone_config)

miniscope_file.gdrive_upload_file(caiman_result_dir,
                                  os.path.join(upload_path, experiment_title, experiment_date, 'caiman', animal_name),
                                  rclone_config)
