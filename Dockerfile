FROM python:3.9.9-slim-buster

WORKDIR /camera-calibration

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --no-cache-dir \
    imageio==2.13.5 \
    numpy==1.21.5 \
    sympy==1.9 \
    opencv-python==4.5.5.62 \
