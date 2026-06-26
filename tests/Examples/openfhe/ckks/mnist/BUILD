load("@heir//tools:heir-openfhe.bzl", "openfhe_lib")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_lib(
    name = "mnist_openfhe",
    generated_lib_header = "mnist_openfhe_lib.inc.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024 modulus-switch-after-mul=true experimental-disable-loop-unroll=true level-budget=40 first-mod-bits=55",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common/mnist:mnist.mlir",
    pybind_target_name = "mnist_openfhe_pybind",
    tags = [
        "nofastbuild",
    ],
)

# TODO(#3114): re-enable after performance issues are figured out
# py_test(
#     name = "mnist_test",
#     srcs = ["mnist_test.py"],
#     data = [
#         "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
#         "@heir//tests/Examples/common/mnist/data:t10k-labels-idx1-ubyte",
#         "@heir//tests/Examples/common/mnist/data:traced_model.pt",
#     ],
#     main = "mnist_test.py",
#     strict_deps = False,
#     tags = [
#         "nofastbuild",  # openfhe is slow unless -c opt
#         "requires-mem:28g",
#     ],
#     deps = [
#         ":mnist_openfhe_pybind",
#         "@abseil-py//absl/testing:absltest",
#         "@heir_pip_deps//numpy",
#         "@heir_pip_deps//torch",
#     ],
# )
