# load("@rules_python//python:py_test.bzl", "py_test")
# load("@heir//tools:heir-openfhe.bzl", "openfhe_lib")

package(default_applicable_licenses = ["@heir//:license"])

# openfhe_lib(
#     name = "lenet_openfhe",
#     generated_lib_header = "lenet_openfhe_lib.inc.h",
#     heir_opt_flags = [
#         "--annotate-module=backend=openfhe scheme=ckks",
#         "--torch-linalg-to-ckks=min-slot-count=1024 greedy-level-budget=15 greedy-modulus-switch-after-mul=true experimental-disable-loop-unroll=true first-mod-bits=30 scaling-mod-bits=24",
#         "--scheme-to-openfhe",
#     ],
#     mlir_src = "lenet.mlir",
#     pybind_target_name = "lenet_openfhe_pybind",
#     tags = [
#         "nofastbuild",
#         "requires-mem:28g",
#     ],
# )
#
# py_test(
#     name = "lenet_test",
#     size = "large",
#     srcs = ["lenet_test.py"],
#     data = [
#         ":lenet.openfhe.mlir",
#         "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
#         "@heir//tests/Examples/common/mnist/data:t10k-labels-idx1-ubyte",
#     ],
#     main = "lenet_test.py",
#     strict_deps = False,
#     tags = [
#         "nofastbuild",  # openfhe is slow unless -c opt
#         "requires-mem:28g",
#     ],
#     deps = [
#         ":lenet_openfhe_pybind",
#         "@abseil-py//absl/testing:absltest",
#         "@heir_pip_deps//numpy",
#         "@heir_pip_deps//torch",
#     ],
# )
