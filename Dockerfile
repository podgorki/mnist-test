FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update

RUN apt-get install gcc
RUN apt-get install git

# remove tiny vim and reinstall full vim
RUN apt-get remove vim -y && apt-get install vim

# Upgrade pip
RUN python3 -m pip install --upgrade pip

WORKDIR /root

ADD "https://api.github.com/repos/podgorki/mnist-test/commits?per_page=1" latest_commit
RUN curl -sLO "https://github.com/podgorki/mnist-test/archive/main.zip" && unzip main.zip
RUN rm main.zip

RUN cd mnist-test-main && pip install -r requirements.txt

ENV DATASETPATH=""
ENV CHECKPOINTPATH=""
ENV LOGPATH=""

ENTRYPOINT ["/bin/bash"]

