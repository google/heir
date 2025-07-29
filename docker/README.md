# Docker for Compiling and Running HEIR

Docker can also be used to compile and run HEIR project from the bleeding edge
source on git. This involves building a Docker image of the development
environment and trigger the build which follows below.

## Building Docker Image

You can build the docker image using command for your platform. Replace
meta-syntactic variables.

### Linux

```
$ docker buildx build -t heir:@imagename@ .
```

## Run Docker Image

Typically we like to map the local HEIR source folder into the container at
/home/heiruser/heir and run it as user heiruser.

To run the docker image follow the commands:

```
$ docker run --user heiruser -v `pwd`/heir:/home/heiruser/heir/  -it heir:@imagename@
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
