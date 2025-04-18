# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "mult_indep_8_debug_test",
    generated_lib_header = "mult_indep_8_debug_lib.h",
    heir_opt_flags = [
        "--mlir-to-secret-arithmetic",
        "--mlir-to-bfv=noise-model=bfv-noise-bmcm23 \
          annotate-noise-bound=true",
        # manual scheme-to-openfhe
        "--bgv-to-lwe",
        "--lwe-add-debug-port",
        "--lwe-to-openfhe",
        "--openfhe-configure-crypto-context=insecure=true ring-dim=32768 mul-depth=6 scaling-mod-size=60",
    ],
    mlir_src = "mult_indep_8.mlir",
    tags = ["notap"],
    test_src = "mult_indep_8_debug_test.cpp",
)
