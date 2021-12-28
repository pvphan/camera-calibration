FROM python:3.9.9-slim-buster

WORKDIR /camera-calibration

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

