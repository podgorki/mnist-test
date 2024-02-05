FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update

RUN apt-get install gcc
RUN apt-get install git

# remove tiny vim and reinstall full vim
RUN apt-get remove vim -y && apt-get install vim

# Upgrade pip
RUN python3 -m pip install --upgrade pip

WORKDIR /root

RUN git clone https://github.com/podgorki/mnist-test.git

RUN cd mnist-test && pip install -r requirements.txt

ENV DATASETPATH=""
ENV CHECKPOINTPATH=""
ENV LOGPATH=""

ENTRYPOINT ["/bin/bash"]

