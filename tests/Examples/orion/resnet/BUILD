load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "resnet10",
    go_library_name = "resnet10",
    heir_opt_flags = [
        "--scheme-to-lattigo",
    ],
    mlir_src = ":resnet10.mlir",
)

go_test(
    name = "resnet10_test",
    srcs = ["resnet10_test.go"],
    embed = [":resnet10"],
)
