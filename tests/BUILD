# Tests for HEIR

load("@heir//bazel:lit.bzl", "lit_test")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Bundle together all of the test utilities that are used by tests.
# this is needed by the lit_test macro in lit.bzl
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@heir//tools:heir-opt",
        "@heir//tools:heir-translate",
        "@heir//tests:lit.cfg.py",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@llvm-project//llvm:count",
        "@llvm-project//mlir:libmlir_c_runner_utils.so",
        "@llvm-project//mlir:libmlir_runner_utils.so",
        "@llvm-project//mlir:mlir-cpu-runner",
        "@llvm-project//mlir:mlir-opt",
    ],
)

lit_test(src="emit_verilog_hello_world.mlir")
lit_test(src="hello_world.tosa.mlir", size="medium")
lit_test(src="memref_copy.mlir")
lit_test(src="memref_copy_multidim.mlir")
lit_test(src="memref_global.mlir")
lit_test(src="memref_global_input_access.mlir")
lit_test(src="memref_global_raw.mlir")
lit_test(src="memref_global_splat.mlir")
lit_test(src="memref_unused.mlir")
lit_test(src="ops.mlir")
