# mnist-test
A basic mnist training script to test k8s functionality

## Building image

When building the docker file overide the empty environment variables with the -e flag

``docker run 
-e DATASETPATH=path/to/dataset
-e CHECKPOINTPATH=path/to/checkpoints 
-e LOGPATH=path/to/checkpoints <imageName>``
