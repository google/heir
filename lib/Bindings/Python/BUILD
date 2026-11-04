load("@llvm-project//mlir:build_defs.bzl", "mlir_c_api_cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_filegroup")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load(":python_package.bzl", "mlir_python_package")

# PROTOTYPE: nanobind-based MLIR python bindings for HEIR, following the layout
# used by upstream's CMake build (and by JAX's bazel build):
#
#   heir_mlir/
#     ir.py, passmanager.py, ...            (upstream python sources)
#     dialects/{func,arith,tensor}.py, ...  (upstream python sources + ODS gen)
#     dialects/secret.py, _secret_ops_gen.py (HEIR ODS gen)
#     _mlir_libs/
#       __init__.py                         (upstream)
#       libHEIRPythonCAPI.so                (ALL of MLIR + HEIR C++, one copy)
#       _mlir.so                            (core nanobind ext, upstream src)
#       _site_initialize_0.so               (HEIR dialect/pass registration)

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

PACKAGE_NAME = "heir_mlir"

# Every nanobind extension (and the recompiled upstream core sources) must agree
# on these, otherwise capsule names / nanobind type names won't match.
cc_library(
    name = "python_defines",
    defines = [
        "MLIR_PYTHON_PACKAGE_PREFIX=%s." % PACKAGE_NAME,
        "MLIR_BINDINGS_PYTHON_DOMAIN=heir",
    ],
)

PY_COPTS = [
    "-fexceptions",
    "-frtti",
]

PY_FEATURES = ["-use_header_modules"]

##---------------------------------------------------------------------------##
# C API (the only interface the python extensions use to talk to the compiler)
##---------------------------------------------------------------------------##

mlir_c_api_cc_library(
    name = "HEIRCAPI",
    srcs = ["HEIRCAPI.cpp"],
    hdrs = ["HEIRCAPI.h"],
    capi_deps = ["@llvm-project//mlir:CAPIIR"],
    header_deps = [],
    deps = [
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/Secret/Transforms",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

# The single shared library holding all compiler code. Everything else links
# against this dynamically so there is exactly one copy of MLIR's global state
# (TypeIDs, pass registry, ...) in the process.
cc_binary(
    name = "libHEIRPythonCAPI.so",
    linkopts = [
        "-Wl,-soname,libHEIRPythonCAPI.so",
        "-Wl,-rpath,$$ORIGIN",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":HEIRCAPIObjects",
        "@llvm-project//mlir:CAPIArithObjects",
        "@llvm-project//mlir:CAPITensorObjects",
        "@llvm-project//mlir:MLIRBindingsPythonCAPIObjects",
    ],
)

cc_library(
    name = "HEIRPythonCAPIShared",
    srcs = [":libHEIRPythonCAPI.so"],
    deps = [
        ":HEIRCAPIHeaders",
        "@llvm-project//mlir:CAPIArithHeaders",
        "@llvm-project//mlir:CAPIAsyncHeaders",
        "@llvm-project//mlir:CAPIDebugHeaders",
        "@llvm-project//mlir:CAPIIRHeaders",
        "@llvm-project//mlir:CAPITensorHeaders",
        "@llvm-project//mlir:CAPITransformsHeaders",
    ],
)

##---------------------------------------------------------------------------##
# Nanobind extensions
##---------------------------------------------------------------------------##

# Upstream's MLIRBindingsPythonCoreNoCAPI, recompiled with our defines.
cc_library(
    name = "HEIRPythonCore",
    srcs = ["@llvm-project//mlir:MLIRBindingsPythonSourceFiles"],
    copts = PY_COPTS,
    features = PY_FEATURES,
    deps = [
        ":HEIRPythonCAPIShared",
        ":python_defines",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:MLIRBindingsPythonLibHeaders",
        "@llvm-project//mlir:MLIRBindingsPythonNanobindHeaders",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:config",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)

cc_binary(
    name = "_mlir.so",
    srcs = ["@llvm-project//mlir:lib/Bindings/Python/MainModule.cpp"],
    copts = PY_COPTS,
    features = PY_FEATURES,
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":HEIRPythonCAPIShared",
        ":HEIRPythonCore",
        ":python_defines",
        "@nanobind",
    ],
)

cc_binary(
    name = "_site_initialize_0.so",
    srcs = ["SiteInitialize.cpp"],
    copts = PY_COPTS,
    features = PY_FEATURES,
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":HEIRPythonCAPIShared",
        ":python_defines",
        "@llvm-project//mlir:MLIRBindingsPythonNanobindHeaders",
        "@nanobind",
    ],
)

##---------------------------------------------------------------------------##
# Python sources
##---------------------------------------------------------------------------##

gentbl_filegroup(
    name = "SecretOpsPyGen",
    tbl_outs = {
        "heir_mlir/dialects/_secret_ops_gen.py": [
            "-gen-python-op-bindings",
            "-bind-dialect=secret",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "@heir//lib/Dialect/Secret/IR:SecretOps.td",
    deps = ["@heir//lib/Dialect/Secret/IR:td_files"],
)

mlir_python_package(
    name = "heir_mlir_pkg",
    package_name = PACKAGE_NAME,
    srcs = [
        "heir_mlir/__init__.py",
        "heir_mlir/dialects/secret.py",
        ":SecretOpsPyGen",
    ],
    native_libs = [
        ":_mlir.so",
        ":_site_initialize_0.so",
        ":libHEIRPythonCAPI.so",
    ],
    upstream_srcs = [
        "@llvm-project//mlir/python:ArithOpsPyFiles",
        "@llvm-project//mlir/python:BuiltinOpsPyFiles",
        "@llvm-project//mlir/python:DialectCorePyFiles",
        "@llvm-project//mlir/python:ExtrasPyFiles",
        "@llvm-project//mlir/python:FuncPyFiles",
        "@llvm-project//mlir/python:IRPyFiles",
        "@llvm-project//mlir/python:MlirLibsPyFiles",
        "@llvm-project//mlir/python:PassManagerPyFiles",
        "@llvm-project//mlir/python:RewritePyFiles",
        "@llvm-project//mlir/python:TensorOpsPyFiles",
    ],
)

py_library(
    name = "heir_mlir",
    data = [":heir_mlir_pkg"],
    imports = ["heir_mlir_pkg"],
)

py_test(
    name = "heir_mlir_test",
    srcs = ["heir_mlir_test.py"],
    deps = [":heir_mlir"],
)
