load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

# Secret arithmetic plaintext backend with original integer type semantic
llvm_runner_test(
    name = "roberts_cross",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=4096",
    ],
    main_c_src = "roberts_cross_test.cpp",
    mlir_src = "@heir//tests/Examples/common:roberts_cross_16x16.mlir",
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
llvm_runner_test(
    name = "roberts_cross_mod_786433",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=4096 plaintext-modulus=786433",
    ],
    main_c_src = "roberts_cross_mod_test.cpp",
    mlir_src = "@heir//tests/Examples/common:roberts_cross_16x16.mlir",
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
llvm_runner_test(
    name = "roberts_cross_mod_65537",
    defines = ["EXPECT_FAILURE"],
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-size=4096 plaintext-modulus=65537",
    ],
    main_c_src = "roberts_cross_mod_test.cpp",
    mlir_src = "@heir//tests/Examples/common:roberts_cross_16x16.mlir",
)
