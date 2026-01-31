load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

llvm_runner_test(
    name = "loop",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=8192",
    ],
    main_c_src = "loop_test.cpp",
    mlir_src = "@heir//tests/Examples/common:loop.mlir",
)

