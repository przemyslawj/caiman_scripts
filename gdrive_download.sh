#!/bin/bash
EXP_MONTH=2019-08
EXP_TITLE=habituation
EXP_DATE=2019-08-27
ANIMAL=E-BL
DOWNSAMPLE=2
TRIAL_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/trial/mv_caimg/${ANIMAL}/"
HOME_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/homecage/mv_caimg/${ANIMAL}/"
TEST_REL_DIR="${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}/test/mv_caimg/${ANIMAL}/"

RCLONE_CONFIG=drive_synced
rclone copy -P ${RCLONE_CONFIG}:cheeseboard/${HOME_REL_DIR} /mnt/DATA/Prez/cheeseboard-down/down_${DOWNSAMPLE}/${HOME_REL_DIR}
rclone copy -P ${RCLONE_CONFIG}:cheeseboard/${TRIAL_REL_DIR} /mnt/DATA/Prez/cheeseboard-down/down_${DOWNSAMPLE}/${TRIAL_REL_DIR}
rclone copy -P ${RCLONE_CONFIG}:cheeseboard/${TEST_REL_DIR} /mnt/DATA/Prez/cheeseboard-down/down_${DOWNSAMPLE}/${TEST_REL_DIR}
