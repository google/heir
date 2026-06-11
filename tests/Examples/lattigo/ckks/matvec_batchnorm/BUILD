load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "matvec_batchnorm",
    go_library_name = "matvecbatchnorm",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--linalg-canonicalizations",
        "--torch-linalg-to-ckks=ciphertext-degree=2048",
        "--scheme-to-lattigo",
    ],
    mlir_src = "matvec_batchnorm.mlir",
    split_preprocessing = False,
)

go_test(
    name = "matvec_batchnorm_test",
    srcs = ["matvec_batchnorm_test.go"],
    embed = [":matvecbatchnorm"],
)
