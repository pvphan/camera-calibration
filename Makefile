
SHELL:=/bin/bash

OUTPUT_PATH=$(HOME)/Documents/camera-calibration-output
REPO_PATH=$(HOME)/git/camera-calibration
WORKDIR_PATH=/camera-calibration
IMAGE_TAG=pvphan/camera-calibration:0.1

test: image
	docker run --rm -it \
		--volume=${REPO_PATH}:${WORKDIR_PATH}:ro \
		${IMAGE_TAG} python3 -m unittest discover tests/

shell: image
	docker run --rm -it \
		--volume=${REPO_PATH}:${WORKDIR_PATH}:ro \
		--volume=${OUTPUT_PATH}:/tmp/output \
		${IMAGE_TAG} bash

image:
	docker build --tag ${IMAGE_TAG} .

uploadImage: image
	docker push ${IMAGE_TAG}
