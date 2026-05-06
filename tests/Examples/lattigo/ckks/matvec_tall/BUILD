load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "matvec_tall",
    go_library_name = "matvectall",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:matvec_8x4.mlir",
)

go_test(
    name = "matvectall_test",
    size = "small",
    srcs = ["matvec_tall_test.go"],
    embed = [":matvectall"],
)
