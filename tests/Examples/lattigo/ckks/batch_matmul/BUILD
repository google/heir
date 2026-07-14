load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "batch_matmul",
    go_library_name = "batchmatmul",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 experimental-disable-loop-unroll=true level-budget=40 first-mod-bits=60 scaling-mod-bits=50",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:batch_matmul.mlir",
)

# TODO(#3089): enable it once the flakiness of bicyclic layout is resolved
# go_test(
#     name = "batch_matmul_test",
#     srcs = ["batch_matmul_test.go"],
#     embed = [":batchmatmul"],
# )

glaze_ignore = ["batch_matmul_test.go"]  # @unused
