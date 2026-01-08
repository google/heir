load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@rules_go//go:def.bzl", "go_test")
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
    name = "mlp_binary",
    srcs = ["mlp_test.cpp"],
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

heir_opt(
    name = "mlp_mlir",
    src = "mlp.dacapo.mlir",
    generated_filename = "mlp.openfhe.mlir",
    pass_flags = [
        "--scheme-to-openfhe",
    ],
)

heir_lattigo_lib(
    name = "mlp",
    go_library_name = "mlp",
    heir_opt_flags = [
        "--scheme-to-lattigo",
    ],
    mlir_src = ":mlp.dacapo.mlir",
)

go_test(
    name = "mlp_test",
    srcs = ["mlp_test.go"],
    embed = [":mlp"],
    deps = [
        "@com_github_tuneinsight_lattigo_v6//circuits/ckks/lintrans",
        "@com_github_tuneinsight_lattigo_v6//core/rlwe",
        "@com_github_tuneinsight_lattigo_v6//ring",
        "@com_github_tuneinsight_lattigo_v6//schemes/ckks",
    ],
)
