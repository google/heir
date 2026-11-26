load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# avgpool1d.mlir is the torch-mlir export of scripts/torch_ci/models/avgpool1d.py,
# which lowers AvgPool1d as a 2D pool over an expanded unit dimension.
heir_lattigo_lib(
    name = "avgpool1d",
    go_library_name = "avgpool1d",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "avgpool1d.mlir",
)

go_test(
    name = "avgpool1d_test",
    size = "small",
    srcs = ["avgpool1d_test.go"],
    embed = [":avgpool1d"],
)
