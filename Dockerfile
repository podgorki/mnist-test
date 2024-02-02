FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && apt-get git

RUN git clone https://github.com/podgorki/mnist-test.git

RUN pip install -r requirements.txt

ENV DATASETPATH=""
ENV CHECKPOINTPATH=""
ENV LOGPATH=""

ENTRYPOINT ["/bin/bash", "-c", "echo", "hello mnist!"]
