load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# elu.mlir is the torch-mlir export of scripts/torch_ci/models/elu.py:
# max(0, x) + exp(min(0, x)) - 1. max and min are approximated by a polynomial
# on the default domain [-1, 1].
heir_lattigo_lib(
    name = "elu",
    go_library_name = "elu",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "elu.mlir",
)

go_test(
    name = "elu_test",
    size = "small",
    srcs = ["elu_test.go"],
    embed = [":elu"],
)
