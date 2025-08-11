# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "mult_indep_32_debug",
    extra_srcs = ["@heir//tests/Examples/lattigo/bfv:bfv_debug.go"],
    go_library_name = "main",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=bfv",
        "--mlir-to-bfv=noise-model=bfv-noise-bmcm23 annotate-noise-bound=true",
        "--scheme-to-lattigo=insert-debug-handler-calls=true",
    ],
    mlir_src = "@heir//tests/Examples/common:mult_indep_32.mlir",
    tags = ["e2e"],
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "multindep32debug_test",
    srcs = ["mult_indep_32_debug_test.go"],
    embed = [":main"],
    tags = ["e2e"],
)
