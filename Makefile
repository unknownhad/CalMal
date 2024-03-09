build:
	@docker build . -t malware-detect:latest

shell: build
	@docker run --runtime nvidia -it --rm -v ${PWD}:/Malware-Detection malware-detect:latest bash

.PHONY: build shell
