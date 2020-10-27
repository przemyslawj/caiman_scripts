#!/bin/bash
export NCORES=4
export DOWNSAMPLE=2
export VID_PREFIX=''

export TRIAL_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/trial/${ANIMAL}/"
export HOME_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/homecage/${ANIMAL}/"
export AFTERTEST_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/aftertest/${ANIMAL}/"
export BEFORETEST_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/beforetest/${ANIMAL}/"
export CAIMAN_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/caiman/${ANIMAL}/"

#export RCLONE_CONFIG=workdrive
#export SRC_ROOTDIR=cheeseboard
#export CONFIG_FILE=~/.config/rclone/rclone.conf
export CONFIG_FILE=`pwd`/env/rclone.conf
export LOCAL_ROOTDIR=/home/prez/neurodata
export RCLONE_CONFIG=local_drive
export SRC_ROOTDIR=/media/prez/My\ Passport/cheeseboard
export DOWNSAMPLE_SUBPATH=cheeseboard-down/down_${DOWNSAMPLE}
#export UPLOAD_PATH=cheeseboard-down/down_${DOWNSAMPLE}
export UPLOAD_PATH=/media/prez/My\ Passport/cheeseboard-down/down_${DOWNSAMPLE}

export DATE_ROOTDIR=${LOCAL_ROOTDIR}/${DOWNSAMPLE_SUBPATH}/${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/

