load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

llvm_runner_test(
    name = "bicyclic_matmul_pt_test",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=2048",
    ],
    main_c_src = "bicyclic_matmul_pt_test.cpp",
    mlir_src = "@heir//tests/Examples/common:bicyclic_matmul_pt.mlir",
)
