load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "matvec",
    go_library_name = "matvec",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=8192",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:matvec.mlir",
)

go_test(
    name = "matvec_test",
    srcs = ["matvec_test.go"],
    embed = [":matvec"],
)

go_test(
    name = "matvec_test2",
    srcs = ["matvec_test2.go"],
    embed = [":matvec"],
)
