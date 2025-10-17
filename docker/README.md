# Docker for Compiling and Running HEIR

Docker can also be used to compile and run HEIR project from the bleeding edge
source on git. This involves building a Docker image of the development
environment and trigger the build which follows below.

## Building Docker Image

You can build the docker image using command for your platform. Replace
meta-syntactic variables (e.g. replace `heir:@imagename@` with `heir:dev`).

### Linux

```
$ docker buildx build --platform linux/amd64 -f docker/Dockerfile -t heir:@imagename@ .
```

## Run Docker Image

Typically we like to map the local HEIR source folder into the container at
/home/heiruser/heir and run it as user heiruser.

To run the docker image follow the commands:

```
$ docker run --user heiruser --platform linux/amd64 -v "$(pwd)":/home/heiruser/heir -it heir:@imagename@
```

Alternatively, from the repository root you can run `./docker/run.sh`. The
script builds the image with `docker/Dockerfile` and binds your working tree at
`$(pwd)` directly to `/home/heiruser/heir` so edits inside the container are
mirrored on the host.

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

<!-- mdformat global-off -->
