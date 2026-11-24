load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# prelu.mlir is the torch-mlir export of scripts/torch_ci/models/prelu.py:
# select(x > 0, x, 0.25 * x). It is rewritten as 0.25 * x + 0.75 * relu(x), and
# relu is approximated by a polynomial on the default domain [-1, 1].
heir_lattigo_lib(
    name = "prelu",
    go_library_name = "prelu",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "prelu.mlir",
)

go_test(
    name = "prelu_test",
    size = "small",
    srcs = ["prelu_test.go"],
    embed = [":prelu"],
)
