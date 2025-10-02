load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

llvm_runner_test(
    name = "dot_product_8f",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend",
    ],
    main_c_src = "dot_product_8f_test.cpp",
    mlir_src = "@heir//tests/Examples/common:dot_product_8f.mlir",
)
