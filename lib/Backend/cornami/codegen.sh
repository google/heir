#!/bin/bash -x

bazel run //tools:scifr-opt -- --emit-scifrbool $1
