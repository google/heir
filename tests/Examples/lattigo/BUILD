# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "lattigo_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

lattigo_end_to_end_test(
    name = "binops_test",
    heir_opt_flags = [
        "--secretize",
        "--mlir-to-secret-arithmetic",
        "--secret-insert-mgmt-bgv",
        "--secret-distribute-generic",
        "--secret-to-bgv=poly-mod-degree=4",
        "--bgv-to-lattigo",
        "--cse",
    ],
    mlir_src = "binops.mlir",
    test_src = "binops_test.go",
)
