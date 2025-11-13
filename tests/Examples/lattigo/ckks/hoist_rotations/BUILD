load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "hoist_rotations",
    go_library_name = "hoistrotations",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=min-slot-count=2048",
        "--scheme-to-lattigo",
    ],
    mlir_src = "hoist_rotations.mlir",
)

go_test(
    name = "hoistrotations_test",
    srcs = ["hoist_rotations_test.go"],
    embed = [":hoistrotations"],
)
