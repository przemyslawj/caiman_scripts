import argparse
import json
import logging
import os
import shutil
import subprocess

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', type=str, help='Experiment name')
parser.add_argument('--dates', nargs='+', help='Experiment dates', default=[])
parser.add_argument('--animals', nargs='+', help='Animals', default=[])
parser.add_argument('--rm_tmp_dir', help='Remove files from local drive after running',
                    action='store_true')
parser.add_argument('--rm_noise',
                    action='store_true',
                    help='Remove noise in miniscope v4 videos: horizontal scanning bars + fluorescence fluctuation',)
parser.add_argument('--filter_frames',
                    action='store_true',
                    help='Filter dark frames from the videos (caused by intermittent miniscope connection issues)')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--start_motion_correct', help='Start pipeline at motion correct stage',
                    action='store_true')

args = parser.parse_args()


with open('pipeline_setup.json') as json_file:
    pipeline_setup = json.load(json_file)


def run_python_cmd(script):
    cp = subprocess.run(['python', script])
    if cp.returncode != 0:
        logging.error('Script %s failed', script)
        if cp.stderr is not None:
            logging.error('Error: %s', str(cp.stderr))
        raise SystemExit


def run(experiment_title, experiment_date, animal):
    os.environ['ANIMAL'] = animal
    os.environ['EXP_DATE'] = exp_date
    os.environ['EXP_TITLE'] = experiment_title
    logging.info('Running pipeline for exp_title=%s, exp_date=%s, animal=%s', experiment_title, experiment_date, animal)

    if args.dry_run:
        return

    if not args.start_motion_correct:
        run_python_cmd('gdrive_download.py')
        run_python_cmd('downsample.py')
        if args.rm_noise:
            run_python_cmd('remove_noise.py')
        if args.filter_frames:
            run_python_cmd('filter_frames.py')

    run_python_cmd('caiman_mc.py')
    run_python_cmd('memmap_mc_files.py')
    run_python_cmd('create_sessions_info.py')
    run_python_cmd('run_cnmfe.py')
    run_python_cmd('gdrive_upload.py')
    local_miniscope_path = os.path.join(pipeline_setup['localTempDirectory'], pipeline_setup['downsampleSubpath'])
    if args.rm_tmp_dir:
        for exp_name in (pipeline_setup['experimentNames'] + ['caiman']):
            local_dir = os.path.join(local_miniscope_path, experiment_title, experiment_date, exp_name, animal)
            if os.path.isdir(local_dir):
                shutil.rmtree(local_dir)


for exp_date in args.dates:
    for animal in args.animals:
        run(args.exp_title, exp_date, animal)