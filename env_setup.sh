#!/bin/bash
mkdir src && cd src && git clone --depth 1 https://github.com/flatironinstitute/CaImAn.git
conda create -n caiman python=3.6 -c conda-forge caiman
source activate caiman && pip install moviepy tifffile ipyparallel peakutils moviepy
cd src/CaImAn && source activate caiman && pip install .
source activate caiman && caimanmanager.py install

