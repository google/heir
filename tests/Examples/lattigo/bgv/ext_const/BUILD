load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "ext_const",
    go_library_name = "extconst",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=bgv",
        "--scheme-to-lattigo",
        "--ext-const-threshold=2",
    ],
    mlir_src = "ext_const.mlir",
    split_preprocessing = False,
)

go_test(
    name = "ext_const_test",
    srcs = ["ext_const_test.go"],
    embed = [":extconst"],
)
