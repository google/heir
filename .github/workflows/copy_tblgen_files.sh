#!/bin/bash

set -eux
set -o pipefail

# enable ** to glob through subdirectories, disabled by default in bash
shopt -s globstar

SRC_BASE="bazel-bin/include"
DEST_BASE="docs/content/en/docs"

echo "Processing Passes"
for FILE in $SRC_BASE/Conversion/**/*.md
do
  chmod 664 "$FILE"
  mkdir -p "$DEST_BASE/Passes/"
  cp "$FILE" "$DEST_BASE/Passes/"
done

echo "Processing Dialects"
for FILE in $SRC_BASE/Dialect/**/*.md
do
  chmod 664 "$FILE"
  mkdir -p "$DEST_BASE/Dialects/"
  cp "$FILE" "$DEST_BASE/Dialects/"
done
