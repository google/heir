# See README.md for setup required to run these tests

# load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

# TODO(#2047): re-enable when we have improved the layout system
# openfhe_end_to_end_test(
#     name = "halevi_shoup_matvec_test",
#     generated_lib_header = "halevi_shoup_matvec_lib.h",
#     heir_opt_flags = [
#         "--annotate-module=backend=openfhe scheme=ckks",
#         "--mlir-to-ckks=ciphertext-degree=8192",
#         "--scheme-to-openfhe=insert-debug-handler-calls=true",
#     ],
#     heir_translate_flags = [
#         "--openfhe-include-type=source-relative",
#     ],
#     mlir_src = "@heir//tests/Examples/common:matvec.mlir",
#     tags = ["notap"],
#     test_src = "halevi_shoup_matvec_test.cpp",
# )
