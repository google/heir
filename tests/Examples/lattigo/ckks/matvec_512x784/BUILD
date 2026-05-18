load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "matvec_512x784",
    go_library_name = "matvec512x784",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        # TODO(#2960): remove split-preprocessing=0 once split-preprocessing handles loops
        "--mlir-to-ckks=ciphertext-degree=1024 modulus-switch-after-mul=true experimental-disable-loop-unroll=true level-budget=40 first-mod-bits=55 split-preprocessing=0",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:matvec_512x784.mlir",
    split_preprocessing = False,
)

go_test(
    name = "matvec512x784_test",
    size = "enormous",
    timeout = "eternal",
    srcs = ["matvec_512x784_test.go"],
    embed = [":matvec512x784"],
    exec_properties = {"mem": "28g"},
)
