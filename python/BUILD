load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "_mlir",
    srcs = [
        "@llvm-project//mlir:lib/Bindings/Python/MainModule.cpp",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:MLIRBindingsPythonCore",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@llvm-project//mlir:Support",
    ],
)

py_library(
    name = "mlir_python_bindings",
    srcs = [
        "@llvm-project//mlir/python:MlirLibsPyFiles",
    ],
    data = [
        ":_mlir.so",
    ],
    srcs_version = "PY3",
)

py_binary(
    name = "hello_mlir",
    srcs = [
        "hello_mlir.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":mlir_python_bindings",
    ],
)
