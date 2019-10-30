FROM continuumio/anaconda3

RUN mkdir /src
WORKDIR /src
RUN apt-get update && apt-get install -y git vim tree

ENV PATH /opt/conda/bin:$PATH
# Workaround to allow using 'conda activate': https://github.com/ContinuumIO/docker-images/issues/89
ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN conda init bash

# Workaround for https://github.com/ContinuumIO/docker-images/issues/49#issuecomment-311556456
RUN apt install -y libgl1-mesa-glx
RUN wget https://downloads.rclone.org/v1.49.3/rclone-v1.49.3-linux-amd64.deb
RUN dpkg -i rclone-v1.49.3-linux-amd64.deb
RUN conda config --set always_yes yes
RUN conda update --yes conda
RUN conda install -c anaconda cython
RUN conda create -n caiman -c conda-forge caiman
RUN conda install -n caiman -c conda-forge moviepy tifffile ipyparallel peakutils moviepy sk-video
RUN conda activate caiman && pip install imageio==2.4.1

RUN git clone --depth 1 https://github.com/flatironinstitute/CaImAn.git
RUN cd CaImAn && conda activate caiman && caimanmanager.py install

RUN git clone --depth 1 https://github.com/przemyslawj/caiman_scripts.git
COPY rclone.conf caiman_scripts/env/rclone.conf

WORKDIR /src/caiman_scripts
#ENTRYPOINT ["./pipeline.sh"]

