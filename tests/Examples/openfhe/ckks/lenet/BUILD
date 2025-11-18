load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_interpreter_test")

# See README.md for setup required to run these tests
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_interpreter_test(
    name = "lenet_test",
    timeout = "eternal",
    generated_heir_opt_filename = "lenet_heir_opt.mlir",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024",
        "--scheme-to-openfhe",
    ],
    mlir_src = "lenet.mlir",
    tags = [
        "manual",
        "notap",
    ],
    test_src = "lenet_test.cpp",
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_binary(
    name = "lenet_binary",
    srcs = ["lenet_main.cpp"],
    data = [
        "lenet.openfhe.mlir",
    ],
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
