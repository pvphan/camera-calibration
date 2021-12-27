FROM python:3.9.9-slim-buster

WORKDIR /camera-calibration

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

