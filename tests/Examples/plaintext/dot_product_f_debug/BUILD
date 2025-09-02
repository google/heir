load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_applicable_licenses = ["@heir//:license"])

cc_library(
    name = "dot_product_8f_debug_test",
    srcs = ["dot_product_8f_debug_test.c"],
)

llvm_runner_test(
    name = "dot_product_8f_debug",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=8 insert-debug-handler-calls=true",
    ],
    # log is visible to crypto backend
    log_file_name = "dot_product_8f_debug.log",
    log_file_visibility = ["@heir//tests/Examples:__subpackages__"],
    mlir_src = "@heir//tests/Examples/common:dot_product_8f.mlir",
    deps = [
        ":dot_product_8f_debug_test",
        "@heir//tests/llvm_runner:memrefCopy",
    ],
)
