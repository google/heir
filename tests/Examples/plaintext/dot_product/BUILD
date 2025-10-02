load("@heir//tests/llvm_runner:llvm_runner.bzl", "llvm_runner_test")

package(default_applicable_licenses = ["@heir//:license"])

# Secret arithmetic plaintext backend with original integer type semantic
llvm_runner_test(
    name = "dot_product_8",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend",
    ],
    main_c_src = "dot_product_8_test.cpp",
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
llvm_runner_test(
    name = "dot_product_8_mod_257",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-modulus=257",
    ],
    main_c_src = "dot_product_8_mod_test.cpp",
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
llvm_runner_test(
    name = "dot_product_8_mod_65537",
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-modulus=65537",
    ],
    main_c_src = "dot_product_8_mod_test.cpp",
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
)

# Secret arithmetic plaintext backend with plaintext modulus semantic
llvm_runner_test(
    name = "dot_product_8_mod_17",
    defines = ["EXPECT_FAILURE"],
    heir_opt_flags = [
        "--mlir-to-plaintext-backend=plaintext-modulus=17",
    ],
    main_c_src = "dot_product_8_mod_test.cpp",
    mlir_src = "@heir//tests/Examples/common:dot_product_8.mlir",
)
