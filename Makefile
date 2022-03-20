WORKDIR_PATH=/camera-calibration
REPO_PATH:=$(dir $(abspath $(firstword $(MAKEFILE_LIST))))
OUTPUT_PATH=${REPO_PATH}/output
IMAGE_TAG=pvphan/camera-calibration:0.1
RUN_FLAGS = \
	--rm \
	--volume=${REPO_PATH}:${WORKDIR_PATH}:ro \
	--volume=${OUTPUT_PATH}:/tmp/output \
	${IMAGE_TAG}


animation: image
	docker run ${RUN_FLAGS} \
		python3 src/animate.py

test: image
	docker run ${RUN_FLAGS} \
		python3 -m unittest discover -s tests/

shell: image
	docker run -it ${RUN_FLAGS} \
		bash

image:
	docker build --tag ${IMAGE_TAG} .

