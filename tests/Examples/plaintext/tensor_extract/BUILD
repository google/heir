load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

# Secret arithmetic plaintext backend with original integer type semantic
llvm_runner_test(
    name = "tensor_extract",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend",
    ],
    main_c_src = "tensor_extract_test.cpp",
    mlir_src = "tensor_extract.mlir",
)
