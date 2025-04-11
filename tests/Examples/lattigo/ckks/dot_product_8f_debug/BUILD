# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "dot_product_8f_debug",
    data = [
        "@heir//tests/Examples/plaintext/dot_product_f_debug:dot_product_8f_debug.log",
    ],
    extra_srcs = ["@heir//tests/Examples/lattigo/ckks:ckks_debug.go"],
    go_library_name = "main",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=1024 \
          encryption-technique-extended=true \
          plaintext-execution-result-file-name=$(location @heir//tests/Examples/plaintext/dot_product_f_debug:dot_product_8f_debug.log)",
        "--scheme-to-lattigo=insert-debug-handler-calls=true",
    ],
    mlir_src = "@heir//tests/Examples/common:dot_product_8f.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

go_test(
    name = "dotproduct8fdebug_test",
    srcs = ["dot_product_8f_debug_test.go"],
    embed = [":main"],
)
