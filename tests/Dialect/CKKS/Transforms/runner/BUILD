load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")
load("@heir//tools:heir-opt.bzl", "heir_opt")
load("@heir//tools:llc.bzl", "llc")
load("@heir//tools:mlir-translate.bzl", "mlir_translate")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_binary.bzl", "py_binary")

package(default_applicable_licenses = ["@heir//:license"])

py_binary(
    name = "generate_relinearize_test_data",
    srcs = ["generate_relinearize_test_data.py"],
)

genrule(
    name = "relinearize_test_data",
    outs = ["relinearize_test_data.h"],
    cmd = "$(execpath :generate_relinearize_test_data) $@",
    tools = [":generate_relinearize_test_data"],
)

heir_opt(
    name = "ntt_helpers_heir_opt",
    src = "ntt_helpers.mlir",
    generated_filename = "ntt_helpers_heir_opt.mlir",
    pass_flags = [
        "--attach-ntt-roots",
        "--heir-polynomial-to-llvm",
    ],
)

mlir_translate(
    name = "ntt_helpers_mlir_translate",
    src = "ntt_helpers_heir_opt.mlir",
    generated_filename = "ntt_helpers_mlir_translate.ll",
    pass_flags = ["--mlir-to-llvmir"],
)

llc(
    name = "ntt_helpers_llc",
    src = "ntt_helpers_mlir_translate.ll",
    generated_filename = "ntt_helpers_llc.o",
    pass_flags = [
        "-filetype=obj",
        "-relocation-model=pic",
    ],
)

cc_library(
    name = "runner_test_support",
    srcs = ["ntt_helpers_llc.o"],
    hdrs = ["relinearize_test_data.h"],
    alwayslink = True,
)

llvm_runner_test(
    name = "relinearize",
    heir_opt_flags = [
        "--ckks-to-polynomial",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "relinearize_test.cc",
    mlir_src = "relinearize.mlir",
    deps = [":runner_test_support"],
)

llvm_runner_test(
    name = "mul_relinearize",
    heir_opt_flags = [
        "--ckks-to-polynomial",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "mul_relinearize_test.cc",
    mlir_src = "mul_relinearize.mlir",
    deps = [":runner_test_support"],
)
