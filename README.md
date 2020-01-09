# Caiman scripts
Python and bash scripts for running Ca imaging pipeline on the miniscope data.
The output are h5 files with the signal: fluorescence traces for the cells extracted from videos.

The pipeline orchestrates several steps:
* Downloading vidoes from Google Drive,
* Spatially downsampling the videos
* motion correcting the videos by caiman
* running the signal extraction by caiman
* postprocessing: cell selection and registration

Extraction of the signal is performed on videos concatenated from the same day.
The directury structure is organized by date and animal.



# Setup
Use docker or set up conda instance. For the list of commands look at scripts
in the env/ directory

# Running the pipeline
1. Extract components with CaImAn by running caiman_scripts
1.0 Prepare data directories, assumes directory structure like in the example below:

├── 2019-08
│   ├── habituation
│   │   ├── 2019-08-27
│   │   │   ├── homecage
│   │   │   │   └── mv_caimg
│   │   │   │       ├── E-BL
│   │   │   │       │   ├── Session1
│   │   │   │       │   │   └── H13_M43_S35
│   │   │   │       │   │       ├── msCam1.avi
│   │   │   │       │   │       ├── ...
│   │   │   │       │   └── Session2
│   │   │   │       │       └── H13_M54_S44
│   │   │   │       │   │       ├── ...
│   │   │   │       ├── E-TR
│   │   │   │       │   ├── Session1
│   │   │   ├── test
│   │   │   │   ├── mv_caimg
│   │   │   │   │   ├── .....
│   │   │   │
│   │   │   └── trial
│   │   │       ├── locations.csv
│   │   │       ├── movie
│   │   │       │   ├── 2019-08-27_E-BL_trial_1.avi
│   │   │       │   ├── tracking
│   │   │       │   │   ├── 2019-08-27_E-BL_trial_1_positions.csv
│   │   │       │   │   ├── 2019-08-27_E-BL_trial_1_positions.csv
│   │   │       │   ├── ...
│   │   │       └── mv_caimg
│   │   │           ├── E-BL
│   │   │           │   ├── Session1
│   │   │           │   │   └── H13_M43_S35
│   │   ├── 2019-08-28
│   │   │   ├── homecage
│   │   │   ...
│   └── learning
│       ├── 2019-08-30
│       │   ├── homecage
│       │   │      ├── movie

1.1 Run the extraction
1.1.1 Update variables in vars_setup.sh
1.1.2 Run the pipeline:
./pipeline.sh --animals 'E-BL E-TL' --dates '2019-08-27' --exp_title habituation --exp_month 2019-08

This creates the files

2. Filter components by running postprocess_components.py
2.1 Update variables in vars_setup.sh (possibly done in 1.1.1)
2.2 Run:
  source vars_setup.sh
2.3 Run postprocessing
	conda activate caiman
	python postprocess_components.py

This should upload results to gdrive.

3. Run the tracking with DeepLabCut
3.1 Run DLC: download movies, create file list, run, upload tracking to gdrive
3.2 Run python openv_mouse_tracker to process DLC result, upload the tracking

4. Merge tracking and extracted calcium traces using matlab code in cheeseboard_analysis repository
update and run cheeseboard_analysis/matlab/runCaimanForExp.m
Upload filtered results to gdrive
