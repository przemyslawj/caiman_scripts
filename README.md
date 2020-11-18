# Caiman scripts
Python scripts for running Ca imaging pipeline on the miniscope data.
The output are h5 files with the signal: fluorescence traces for the cells extracted from videos.

The pipeline orchestrates several steps:
1) Downloading videos from external storage (e.g. external drive, Google Drive)
2) Filtering video frames and removing noise from the recording (optional)
2) Spatially downsampling the videos
3) Motion correcting the videos by caiman
4) Running the signal extraction by caiman
5) Postprocessing: cell selection and registration
6) Uploading videos to external storage

Extraction of the signal is performed on videos concatenated from the same day.
The directory structure is defined in the `pipeline_setup.json`. The
definition has a format as in userConfig.json used by Miniscope DAQ software.
To allow concatenation of the same day videos, the recordings need to placed
under directories specific for date and animalID.

# Setup
Use docker or set up conda environment. For commands helping with setting up the
caiman instance look at scripts in the env/ directory.
Syncing with external storage uses rclone tool[https://rclone.org/]. Rclone needs
to be configured before running the pipeline. The configuration points to
external storage from which to copy the input videos to a temporary directory
and to which upload the result files. The external storage can be for example
Google Drive, external drive or a local location.
Create your rclone configuration and copy it to file to `env/rclone.conf` or
update the path and configuration name in `pipeline_setup.json`.

# Ca imaging analysis pipeline
## Extract components with CaImAn
0.1. Prepare data directories, an example directory structure and its
definition could look like in the example below:

Definition in `pipeline_setup.json`:
```
"sourceDirectory": "/directory-with-experiment-data/",
"directoryStructure": [
  "experimentGroup",
  "date",
  "experimentName",
  "caimgDirectory",
  "animalID",
  "session",
  "time"
],
```

Directory strucutre
```
/directory-with-experiment-data/
├── habituation
│   ├── 2019-08-27
│   │   ├── homecage
│   │   │   └── mv_caimg
│   │   │       ├── E-BL
│   │   │       │   ├── Session1
│   │   │       │   │   └── H13_M43_S35
│   │   │       │   │       ├── msCam1.avi
│   │   │       │   │       ├── ...
│   │   │       │   └── Session2
│   │   │       │       └── H13_M54_S44
│   │   │       │   │       ├── ...
│   │   │       ├── E-TR
│   │   │       │   ├── Session1
│   │   ├── aftertest
│   │   │   ├── mv_caimg
│   │   │   │   ├── .....
│   │   │
│   │   └── trial
│   │       ├── locations.csv
│   │       ├── movie
│   │       │   ├── 2019-08-27_E-BL_trial_1.avi
│   │       │   ├── tracking
│   │       │   │   ├── 2019-08-27_E-BL_trial_1_positions.csv
│   │       │   │   ├── 2019-08-27_E-BL_trial_1_positions.csv
│   │       │   ├── ...
│   │       └── mv_caimg
│   │           ├── E-BL
│   │           │   ├── Session1
│   │           │   │   └── H13_M43_S35
│   ├── 2019-08-28
│   │   ├── homecage
│   │   ...
└── learning
    ├── 2019-08-30
    │   ├── homecage
    │   │      ├── movie
```

0.1. Prepare file defining how the calcium imaging movies should be cropped.
The csv file needs to have columns: animal, x1, y1, x2, y2.
The coordinates x1,y1 define the upper left corner,
x2, y2 define the bottom right corner of the rectangular ROI. Example file
defining rois is placed in example/rois.csv. The file named `rois.csv` should
be placed directly in the root directory with the source data.

0.2. (optional) Adjust CNMFE source extraction params.
The file defines parameters for the CNMFE algorithm. It defines params:
`gSig`, `gSiz`, `min_corr`, `min_pnr`, `ring_factor` which can be adjusted for
each animal. Example file is placed in `example/cnmfe_params.csv`.
File with the same name should be placed in the experiment month directory
together with `rois.csv`. If not present or there is no definition for the
animal subject, the default params are used which work fine for dCA1.

0.3. Configure the pipeline by editing `pipeline_setup.json`

## Run the pipeline with the source extraction
1.1 Update variables in `vars_setup.sh`
1.2 Run the pipeline:
```
python pipeline.sh E-BL E-TL --dates 2019-08-27 --exp_title habituation
```
For help run `python pipeline.sh -h`

The pipeline creates the h5 files with the extracted signal in subdirectory
`caiman` and uploads the files to external storage.

## Postprocess data
1. Prepare environment. On linux:
```
  export ANIMAL=XX
```
2. Run postprocessing
```
conda activate caiman
python postprocess_components.py
```


## Follow up steps:
### Tracking with DeepLabCut
1. Run DLC: download movies, create file list, run, upload tracking to gdrive
2. Run python `openv_mouse_tracker` to process DLC result, upload the tracking

### Merge tracking and extracted calcium traces
Currently uses matlab code in the private `cheeseboard_analysis` repository.


## Configuration TODOs:
* directory structure requires EXP_TITLE
* configure to allow custom concatenation of videos
* json configuration for postprocessing: registration and components filtering

