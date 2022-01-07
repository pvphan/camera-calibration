FROM python:3.9.9-slim-buster

WORKDIR /camera-calibration

RUN pip install --no-cache-dir \
    imageio==2.13.5 \
    numpy==1.21.5 \
    scipy==1.7.3 \
