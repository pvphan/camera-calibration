
SHELL:=/bin/bash

REPO_PATH=$(HOME)/git/camera-calibration
WORKDIR_PATH=/camera-calibration
IMAGE_TAG=pvphan/camera-calibration:0.1

test: image
	docker run --rm -it \
		--volume=${REPO_PATH}:${WORKDIR_PATH}:ro \
		${IMAGE_TAG} python3 -m unittest discover

shell: image
	docker run --rm -it \
		--volume=${REPO_PATH}:${WORKDIR_PATH}:ro \
		${IMAGE_TAG} bash

image:
	docker build --tag ${IMAGE_TAG} .

uploadImage: image
	docker push ${IMAGE_TAG}
