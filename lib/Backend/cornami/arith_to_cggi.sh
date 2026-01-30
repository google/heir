#!/bin/bash -x


bazel run //tools:heir-opt -- --mlir-to-secret-arithmetic --yosys-optimizer="mode=Boolean" --secret-distribute-generic --secret-to-cggi $1 >$PWD/cornami/tests/cggi_output.mlir
cat $PWD/cornami/tests/cggi_output.mlir

sleep 2
bazel run //tools:heir-opt -- --cggi-to-scifrbool --replace-op-with-section="nChips=2" $PWD/cornami/tests/cggi_output.mlir >$PWD/cornami/tests/scifrbool_output.mlir
cat $PWD/cornami/tests/scifrbool_output.mlir


bazel run --ui_event_filters=-info,-debug,-warning,-stderr,-stdout --noshow_progress --logging=0 //tools:heir-opt -- --view-op-graph $PWD/cornami/tests/cggi_output.mlir >$PWD/cornami/tests/cggi_output.dot
sleep 2
dot -Tpdf $PWD/cornami/tests/cggi_output.dot >$PWD/cornami/tests/cggi_output.pdf

sleep 2
bazel run //tools:heir-translate -- --emit-scifrbool $PWD/cornami/tests/scifrbool_output.mlir >$PWD/cornami/tests/scifrbool_output.cpp
cat $PWD/cornami/tests/scifrbool_output.cpp
