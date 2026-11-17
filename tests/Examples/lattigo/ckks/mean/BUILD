load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# mean.mlir is the torch-mlir export of scripts/torch_ci/models/mean.py:
# torch.mean(x, dim=1), a sum reduction followed by a division by the
# plaintext 64, which becomes a multiplication by 1/64.
heir_lattigo_lib(
    name = "mean",
    go_library_name = "mean",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "mean.mlir",
)

go_test(
    name = "mean_test",
    size = "small",
    srcs = ["mean_test.go"],
    embed = [":mean"],
)
