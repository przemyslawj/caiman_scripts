docker run -i \
    --mount type=bind,source=/mnt/disks/gce-containers-mounts/gce-persistent-disks/disk-1,target=/mnt/DATA \
    -t gcr.io/caiman-252413/caiman_scripts \
    /bin/bash
