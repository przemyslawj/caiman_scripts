#!/bin/bash

conda activate caiman
source vars_setup.sh
./gdrive_download.sh
python downsample.py
#./gdrive_download_processed.sh
time python caiman_mc.py
python memmap_mc_files.py
python create_sessions_info.py
python run_cnmfe.py
./gdrive_upload.sh

