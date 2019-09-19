#!/bin/bash
curl https://downloads.rclone.org/v1.49.3/rclone-v1.49.3-linux-amd64.deb
sudo dpkg -i rclone-v1.49.3-linux-amd64.deb

conda install -c anaconda cython
conda create -n caiman -c conda-forge caiman
#conda activate caiman && pip install moviepy tifffile ipyparallel peakutils moviepy sk-video
conda install -n caiman -c conda-forge moviepy tifffile ipyparallel peakutils moviepy sk-video

mkdir src && cd src && git clone --depth 1 https://github.com/przemyslawj/caiman_scripts.git
cd src && git clone --depth 1 https://github.com/flatironinstitute/CaImAn.git
#cd src/CaImAn && conda activate caiman && pip install -e .
cd src/CaImAn && conda activate caiman && caimanmanager.py install

