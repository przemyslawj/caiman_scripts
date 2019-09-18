#!/bin/bash
mkdir src && cd src && git clone --depth 1 https://github.com/flatironinstitute/CaImAn.git
conda install -c anaconda cython
conda create -n caiman -c conda-forge caiman
conda activate caiman && pip install moviepy tifffile ipyparallel peakutils moviepy
#cd src/CaImAn && conda activate caiman && pip install -e .
conda activate caiman && caimanmanager.py install

