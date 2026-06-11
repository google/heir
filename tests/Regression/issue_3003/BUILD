load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "bug_minrepro",
    go_library_name = "bugminrepro",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024 first-mod-bits=60",
        "--scheme-to-lattigo",
    ],
    mlir_src = "issue_3003.mlir",
)

go_test(
    name = "bugminrepro_test",
    size = "small",
    srcs = ["issue_3003_test.go"],
    embed = [":bugminrepro"],
)
