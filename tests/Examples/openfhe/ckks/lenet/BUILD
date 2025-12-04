# See README.md for setup required to run these tests
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_applicable_licenses = ["@heir//:license"])

# A binary that lets you pass in a lenet.mlir in openfhe dialect
# (pre-compiled) and runs it with timing. For performance profiling,
# use
#
#   bazel build -c opt --copt=-g --linkopt=-lprofiler \
#   tests/Examples/openfhe/ckks/lenet:lenet_binary
#
#   CPUPROFILE=prof.out bazel-bin/tests/Examples/openfhe/ckks/lenet/lenet_binary \
#   tests/Examples/openfhe/ckks/lenet/pre_compiled_lenet.openfhe.mlir
#
# Then a pprof invocation such as
#
#   pprof --text --lines --focus=mlir::heir::openfhe::Interpreter \
#   ./bazel-bin/tests/Examples/openfhe/ckks/lenet/lenet_binary \
#   prof.out > interpreter_focus.txt
cc_binary(
    name = "lenet_binary",
    srcs = ["lenet_main.cpp"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "@bazel_tools//tools/cpp/runfiles",
        "@heir//lib/Target/OpenFhePke:Interpreter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)
