load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# div.mlir is the torch-mlir export of scripts/torch_ci/models/div.py. The
# division of two secrets is computed as x * (1 / y), with 1 / y approximated by
# a Chebyshev initial guess and Goldschmidt iterations on the default divisor
# domain [0.1, 2.0].
heir_lattigo_lib(
    name = "div",
    go_library_name = "div",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "div.mlir",
)

go_test(
    name = "div_test",
    size = "small",
    srcs = ["div_test.go"],
    embed = [":div"],
)
