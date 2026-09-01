load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# End-to-end coverage for the composite-sign ReLU approximation
# (--use-composite-relu): x * step(x/B) built from three chained minimax
# Chebyshev polynomials, with B taken from the calibrated domain attrs.
#
# The input is in the shape the torch importer produces (a linalg.generic
# carrying the domain, cmpf+select in its body), so this also covers
# activation-canonicalizations forwarding domain_lower/domain_upper down onto
# the arith.maximumf that the approximation reads. Without that forwarding the
# approximation would silently fall back to [-1, 1], pick the wrong B, and the
# decrypted result would be wrong outside the unit interval -- which the
# assertions below would catch.
heir_lattigo_lib(
    name = "relu_composite",
    go_library_name = "relucomposite",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=min-slot-count=1024 use-composite-relu=true greedy-level-budget=14",
        "--scheme-to-lattigo",
    ],
    mlir_src = "@heir//tests/Examples/common:relu_composite.mlir",
    split_preprocessing = False,
)

go_test(
    name = "relucomposite_test",
    srcs = ["relu_composite_test.go"],
    embed = [":relucomposite"],
)
