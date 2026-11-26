load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# gelu.mlir is the torch-mlir export of scripts/torch_ci/models/gelu.py:
# x * (1 + erf(x / sqrt(2))) / 2. The division by sqrt(2) becomes a plaintext
# multiplication, and erf is approximated by a polynomial on the default domain
# [-1, 1].
heir_lattigo_lib(
    name = "gelu",
    go_library_name = "gelu",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "gelu.mlir",
)

go_test(
    name = "gelu_test",
    size = "small",
    srcs = ["gelu_test.go"],
    embed = [":gelu"],
)
