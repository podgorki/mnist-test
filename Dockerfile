FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && apt-get git

RUN git clone https://github.com/podgorki/mnist-test.git

ENTRYPOINT ["/bin/bash", "-c", "echo", "hello mnist!"]
