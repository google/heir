load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# maxpool1d.mlir is the torch-mlir export of scripts/torch_ci/models/maxpool1d.py,
# which lowers MaxPool1d as a 2D pool over an expanded unit dimension. The max
# is approximated by a polynomial on the default domain [-1, 1].
heir_lattigo_lib(
    name = "maxpool1d",
    go_library_name = "maxpool1d",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "maxpool1d.mlir",
)

go_test(
    name = "maxpool1d_test",
    size = "small",
    srcs = ["maxpool1d_test.go"],
    embed = [":maxpool1d"],
)
