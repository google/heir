load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "in_place",
    go_library_name = "main",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "in_place.mlir",
)

go_test(
    name = "in_place_test",
    srcs = ["in_place_test.go"],
    embed = [":main"],
)
