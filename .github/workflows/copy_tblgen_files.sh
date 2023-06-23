#!/bin/bash

set -eux
set -o pipefail

# enable ** to glob through subdirectories, disabled by default in bash
shopt -s globstar

SRC_BASE="bazel-bin/include"
DEST_BASE="docs/content/en/docs"

echo "Processing Passes"
mkdir -p "$DEST_BASE/Passes/"
for SRC_PATH in $SRC_BASE/Conversion/**/*.md
do
  FILENAME=$(basename "$SRC_PATH")
  FILENAME_BASE=$(basename "$SRC_PATH" .md)
  DEST_PATH="$DEST_BASE/Passes/$FILENAME"
  cat <<EOF > "$DEST_PATH"
---
title: $FILENAME_BASE
github_url: https://github.com/google/heir/edit/main/$SRC_PATH
---
EOF
  cat "$SRC_PATH" >> "$DEST_PATH"
done

echo "Processing Dialects"
mkdir -p "$DEST_BASE/Dialects/"
for SRC_PATH in $SRC_BASE/Dialect/**/*.md
do
  FILENAME=$(basename "$SRC_PATH")
  FILENAME_BASE=$(basename "$SRC_PATH" .md)
  DEST_PATH="$DEST_BASE/Dialects/$FILENAME"
  cat <<EOF > "$DEST_PATH"
---
title: $FILENAME_BASE
github_url: https://github.com/google/heir/edit/main/$SRC_PATH
---
EOF
  cat "$SRC_PATH" >> "$DEST_PATH"
done

