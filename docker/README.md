# Docker for Compiling and Running HEIR
We suggest using Docker to compile and run HEIR project from the bleeding edge source on git. This involves building a Docker image of the development environment and trigger the build which follows below.

## Building Docker Image
You can build the docker image using command for your platform. Replace meta-syntactic variables.
### Linux
```
$ docker buildx build -t heir:@imagename@ .
```

## Run Docker Image
Typically we like to map the local HEIR source folder into the container at /home/ec2user/heir and run it as user ec2user.

To run the docker image follow the commands:
```
$ docker run --user ec2user -v `pwd`/heir:/home/ec2user/heir/  -it heir:@imagename@ 
```

# Compiling HEIR
Within the docker image you run,
```
$ pushd ~/heir
$ bazelisk build //tools:heir-opt //tools:heir-translate
$ popd
```

# Running HEIR
Within the docker image you run,
```
$ pushd ~heir
$ bazelisk run //tools:heir-opt -- --help
$ popd
```
