load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "matvec_512x784",
    go_library_name = "matvec512x784",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:matvec_512x784.mlir",
)

go_test(
    name = "matvec512x784_test",
    size = "enormous",
    srcs = ["matvec_512x784_test.go"],
    embed = [":matvec512x784"],
    exec_properties = {"mem": "28g"},
)
