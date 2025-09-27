load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

# Secret arithmetic plaintext backend with original integer type semantic
llvm_runner_test(
    name = "add",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend",
    ],
    main_c_src = "add_test.cpp",
    mlir_src = "add.mlir",
)
