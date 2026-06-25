load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")

package(default_applicable_licenses = ["@heir//:license"])

heir_lattigo_lib(
    name = "bicyclic_matmul",
    go_library_name = "bicyclicmatmul",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=2048 enable-split-preprocessing=false",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:bicyclic_matmul.mlir",
    split_preprocessing = False,
)

# TODO(#3089): disable until flakiness is resolved
# go_test(
#     name = "bicyclic_matmul_test",
#     srcs = ["bicyclic_matmul_test.go"],
#     embed = [":bicyclicmatmul"],
# )

glaze_ignore = ["bicyclic_matmul_test.go"]  # @unused
