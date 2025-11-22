# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "linear_transform",
    go_library_name = "linear_transform",
    heir_opt_flags = [
        "--scheme-to-lattigo",
    ],
    mlir_src = ":linear_transform.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.
go_test(
    name = "linear_transform_test",
    srcs = ["linear_transform_test.go"],
    embed = [":linear_transform"],
)
