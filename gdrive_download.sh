#!/bin/bash
rclone copy --config ${CONFIG_FILE} ${RCLONE_CONFIG}:cheeseboard/${EXP_MONTH}/rois.csv \
    ${LOCAL_ROOTDIR}/${DOWNSAMPLE_SUBPATH}/${EXP_MONTH}
rclone copy -P --config ${CONFIG_FILE} ${RCLONE_CONFIG}:cheeseboard/${HOME_REL_DIR} \
    ${LOCAL_ROOTDIR}/${DOWNSAMPLE_SUBPATH}/${HOME_REL_DIR}
rclone copy -P --config ${CONFIG_FILE} ${RCLONE_CONFIG}:cheeseboard/${TRIAL_REL_DIR} \
    ${LOCAL_ROOTDIR}/${DOWNSAMPLE_SUBPATH}/${TRIAL_REL_DIR}
rclone copy -P --config ${CONFIG_FILE} ${RCLONE_CONFIG}:cheeseboard/${AFTERTEST_REL_DIR} \
    ${LOCAL_ROOTDIR}/${DOWNSAMPLE_SUBPATH}/${AFTERTEST_REL_DIR}
rclone copy -P --config ${CONFIG_FILE} ${RCLONE_CONFIG}:cheeseboard/${BEFORETEST_REL_DIR} \
    ${LOCAL_ROOTDIR}/${DOWNSAMPLE_SUBPATH}/${BEFORETEST_REL_DIR}
