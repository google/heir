load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_test.bzl", "py_test")

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
        "nofastbuild",
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

heir_opt(
    name = "lenet_mlir_opt",
    src = "lenet.mlir",
    generated_filename = "lenet.openfhe.mlir",
    pass_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024",
        "--scheme-to-openfhe",
    ],
    tags = [
        "nofastbuild",
        "requires-mem:28g",
    ],
)

cc_library(
    name = "interpreter_shim",
    srcs = ["interpreter_shim.cpp"],
    hdrs = ["interpreter_shim.h"],
    deps = [
        "@heir//lib/Target/OpenFhePke:Interpreter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

pybind_extension(
    name = "lenet_interpreter",
    srcs = ["interpreter_bindings.cpp"],
    deps = [
        ":interpreter_shim",
    ],
)

py_test(
    name = "lenet_test",
    timeout = "eternal",
    srcs = ["lenet_test.py"],
    data = [
        ":lenet.openfhe.mlir",
        "@heir//tests/Examples/openfhe/ckks/mnist/data:t10k-images-idx3-ubyte",
        "@heir//tests/Examples/openfhe/ckks/mnist/data:t10k-labels-idx1-ubyte",
    ],
    main = "lenet_test.py",
    strict_deps = False,
    tags = [
        "manual",
        "nofastbuild",  # openfhe is slow unless -c opt
        "notap",
        "requires-mem:28g",
    ],
    deps = [
        ":lenet_interpreter",
        "@abseil-py//absl/testing:absltest",
        "@heir_pip_deps//numpy",
        "@heir_pip_deps//torch",
    ],
)
