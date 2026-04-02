load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

llvm_runner_test(
    name = "lower_convert_basis",
    heir_opt_flags = [
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = "lower_convert_basis_test.cc",
    mlir_src = "lower_convert_basis.mlir",
)
