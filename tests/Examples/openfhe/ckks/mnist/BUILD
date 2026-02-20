load("@heir//tools:heir-openfhe.bzl", "openfhe_lib")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_lib(
    name = "mnist_openfhe",
    # ensure CppCompile has enough memory to compile this large, unrolled program
    exec_properties = {"mem": "28g"},
    generated_lib_header = "mnist_openfhe_lib.inc.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common/mnist:mnist.mlir",
    pybind_target_name = "mnist_openfhe_pybind",
    tags = [
        "nofastbuild",
        "notap",
    ],
)

py_test(
    name = "mnist_test",
    size = "enormous",
    srcs = ["mnist_test.py"],
    data = [
        "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
        "@heir//tests/Examples/common/mnist/data:t10k-labels-idx1-ubyte",
        "@heir//tests/Examples/common/mnist/data:traced_model.pt",
    ],
    main = "mnist_test.py",
    tags = [
        "nofastbuild",  # openfhe is slow unless -c opt
        "requires-mem:28g",
    ],
    deps = [
        ":mnist_openfhe_pybind",
        "@abseil-py//absl/testing:absltest",
        "@heir_pip_deps//numpy",
        "@heir_pip_deps//torch",
    ],
)

openfhe_lib(
    name = "mnist_openfhe_layer1",
    # ensure CppCompile has enough memory to compile this large, unrolled program
    exec_properties = {"mem": "28g"},
    generated_lib_header = "mnist_openfhe_layer1_lib.inc.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024",
        "--scheme-to-openfhe",
    ],
    mlir_src = "@heir//tests/Examples/common/mnist:mnist_layer1.mlir",
    pybind_target_name = "mnist_openfhe_layer1_pybind",
    tags = [
        "nofastbuild",
        "notap",
    ],
)

py_test(
    name = "mnist_test_layer1",
    size = "enormous",
    srcs = ["mnist_test_layer1.py"],
    data = [
        "@heir//tests/Examples/common/mnist/data:t10k-images-idx3-ubyte",
        "@heir//tests/Examples/common/mnist/data:t10k-labels-idx1-ubyte",
        "@heir//tests/Examples/common/mnist/data:traced_model.pt",
    ],
    main = "mnist_test_layer1.py",
    tags = [
        "nofastbuild",  # openfhe is slow unless -c opt
        "requires-mem:28g",
    ],
    deps = [
        ":mnist_openfhe_layer1_pybind",
        "@abseil-py//absl/testing:absltest",
        "@heir_pip_deps//numpy",
        "@heir_pip_deps//torch",
    ],
)
