load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# neg.mlir is the torch-mlir export of scripts/torch_ci/models/neg.py.
heir_lattigo_lib(
    name = "neg",
    go_library_name = "neg",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "neg.mlir",
)

go_test(
    name = "neg_test",
    size = "small",
    srcs = ["neg_test.go"],
    embed = [":neg"],
)
