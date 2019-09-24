#!/bin/bash
rclone copy -P --config ${CONFIG_FILE} ${RCLONE_CONFIG}:cheeseboard/${HOME_REL_DIR} \
    ${LOCAL_ROOTDIR}/${DOWNSAMPLE_SUBPATH}/${HOME_REL_DIR}
rclone copy -P --config ${CONFIG_FILE} ${RCLONE_CONFIG}:cheeseboard/${TRIAL_REL_DIR} \
    ${LOCAL_ROOTDIR}/${DOWNSAMPLE_SUBPATH}/${TRIAL_REL_DIR}
rclone copy -P --config ${CONFIG_FILE} ${RCLONE_CONFIG}:cheeseboard/${TEST_REL_DIR} \
    ${LOCAL_ROOTDIR}/${DOWNSAMPLE_SUBPATH}/${TEST_REL_DIR}
