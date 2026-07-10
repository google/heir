load("@heir//tools:heir-openfhe.bzl", "openfhe_lib")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_lib(
    name = "mnist_rotom_openfhe",
    data = glob(["inputs/*.npz"]),
    exec_properties = {"mem": "28g"},
    generated_lib_header = "mnist_rotom_openfhe_lib.inc.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=32768 scaling-mod-bits=45 first-mod-bits=52",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/openfhe/ckks/rotom/mnist:mnist.mlir",
    pybind_target_name = "mnist_rotom_openfhe_pybind",
    tags = [
        "nofastbuild",
    ],
)

py_test(
    name = "mnist_rotom",
    size = "enormous",
    srcs = ["mnist_test.py"],
    data = [
        "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
        "@heir//tests/Examples/common/mnist/data:t10k-labels-idx1-ubyte",
        "@heir//tests/Examples/common/mnist/data:traced_model.pt",
    ] + glob(["inputs/*.npz"]),
    main = "mnist_test.py",
    tags = [
        "nofastbuild",
        "requires-mem:28g",
    ],
    deps = [
        ":mnist_rotom_openfhe_pybind",
        "@abseil-py//absl/testing:absltest",
        "@heir_pip_deps//numpy",
        "@heir_pip_deps//torch",
    ],
)
