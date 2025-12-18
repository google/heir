# See README.md for setup required to run these tests

load("@heir//tools:heir-openfhe.bzl", "openfhe_lib")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_applicable_licenses = ["@heir//:license"])

# Generate the OpenFHE library with Python bindings
openfhe_lib(
    name = "chained_matmul_openfhe",
    data = glob(["inputs/*.npz"]) + glob(["results/*.npz"]),
    generated_lib_header = "chained_matmul_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=4096 enable-arithmetization=false scaling-mod-bits=45",
        "--scheme-to-openfhe",
    ],
    heir_translate_flags = [],
    mlir_src = "@heir//tests/Examples/openfhe/ckks/rotom/chained_matmul:chained_matmul.mlir",
    pybind_target_name = "chained_matmul_lib",
    tags = ["nofastbuild"],
)

# Python test
py_test(
    name = "chained_matmul_test_py",
    size = "large",
    srcs = ["chained_matmul_test.py"],
    data = glob(["inputs/*.npz"]) + glob(["results/*.npz"]),
    main = "chained_matmul_test.py",
    tags = ["nofastbuild"],
    deps = [
        ":chained_matmul_lib",
        "@abseil-py//absl/testing:absltest",
        "@heir_pip_deps//numpy",
    ],
)
