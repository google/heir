load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# leakyrelu.mlir is the torch-mlir export of scripts/torch_ci/models/leakyrelu.py:
# max(0, x) + 0.01 * min(0, x). max and min are approximated by a polynomial
# on the default domain [-1, 1].
heir_lattigo_lib(
    name = "leakyrelu",
    go_library_name = "leakyrelu",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "leakyrelu.mlir",
)

go_test(
    name = "leakyrelu_test",
    size = "small",
    srcs = ["leakyrelu_test.go"],
    embed = [":leakyrelu"],
)
