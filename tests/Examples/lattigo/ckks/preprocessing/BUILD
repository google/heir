load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "matvec",
    go_library_name = "matvec",
    heir_opt_flags = [
        "--annotate-module=scheme=ckks backend=lattigo",
        "--torch-linalg-to-ckks=split-preprocessing=2",
        "--scheme-to-lattigo",
    ],
    mlir_src = "matvec.mlir",
    split_preprocessing = True,
)

go_test(
    name = "matvec_test",
    srcs = ["matvec_test.go"],
    embed = [":matvec"],
    deps = [":matvec_utils"],
)
