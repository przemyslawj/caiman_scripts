#!/bin/bash
export EXP_MONTH=2019-08
export EXP_TITLE=habituation
export EXP_DATE=2019-08-27
export ANIMAL=E-BL
export DOWNSAMPLE=2
export TRIAL_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/trial/mv_caimg/${ANIMAL}/"
export HOME_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/homecage/mv_caimg/${ANIMAL}/"
export TEST_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/test/mv_caimg/${ANIMAL}/"
export CAIMAN_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/caiman/${ANIMAL}/"
export NCORES=8

#export RCLONE_CONFIG=drive_synced
export RCLONE_CONFIG=workdrive
#export CONFIG_FILE=~/.config/rclone/rclone.conf
export CONFIG_FILE=`pwd`/env/rclone.conf
export LOCAL_ROOTDIR=/mnt/DATA/Prez
#export LOCAL_ROOTDIR=/home/przemek/neurodata
export DOWNSAMPLE_SUBPATH=cheeseboard-down/down_${DOWNSAMPLE}

