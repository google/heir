load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load(":python_package.bzl", "heir_py_dialect", "mlir_python_package")

# The `heir` python package (published to PyPI as `heir_py`): HEIR's MLIR
# python bindings plus a thin convenience layer. The native code lives in
# //lib/Bindings/Python.

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# dialect name -> (top-level ops .td file under lib/Dialect, whether the dialect
# defines enums, td_library deps besides the dialect's own :td_files)
HEIR_DIALECTS = {
    "bgv": (
        "BGV/IR/BGVOps.td",
        False,
        [
            "@heir//lib/Dialect/LWE/IR:td_files",
            "@heir//lib/Dialect/Polynomial/IR:td_files",
            "@heir//lib/Dialect:td_files",
        ],
    ),
    "cggi": (
        "CGGI/IR/CGGIOps.td",
        False,
        [
            "@heir//lib/Dialect/LWE/IR:td_files",
            "@heir//lib/Dialect/Polynomial/IR:td_files",
            "@heir//lib/Dialect:td_files",
        ],
    ),
    "cheddar": (
        "Cheddar/IR/CheddarOps.td",
        False,
        ["@heir//lib/Dialect:td_files"],
    ),
    "ckks": (
        "CKKS/IR/CKKSOps.td",
        False,
        [
            "@heir//lib/Dialect/LWE/IR:td_files",
            "@heir//lib/Dialect/Polynomial/IR:td_files",
            "@heir//lib/Dialect:td_files",
        ],
    ),
    "comb": ("Comb/IR/Comb.td", True, []),
    "debug": ("Debug/IR/DebugOps.td", False, []),
    "jaxite": (
        "Jaxite/IR/JaxiteOps.td",
        False,
        ["@heir//lib/Dialect/LWE/IR:td_files"],
    ),
    "jaxiteword": (
        "JaxiteWord/IR/JaxiteWordOps.td",
        False,
        ["@heir//lib/Dialect/LWE/IR:td_files"],
    ),
    "kernel": (
        "Kernel/IR/KernelOps.td",
        False,
        ["@heir//lib/Dialect:td_files"],
    ),
    "key_mgmt": ("KeyMgmt/IR/KeyMgmtOps.td", False, []),
    "lattigo": (
        "Lattigo/IR/LattigoOps.td",
        False,
        ["@heir//lib/Dialect:td_files"],
    ),
    "lwe": (
        "LWE/IR/LWEOps.td",
        True,
        ["@heir//lib/Dialect/Polynomial/IR:td_files"],
    ),
    "math_ext": ("MathExt/IR/MathExtOps.td", False, []),
    "mgmt": ("Mgmt/IR/MgmtOps.td", False, []),
    "mod_arith": ("ModArith/IR/ModArithOps.td", True, []),
    "openfhe": (
        "Openfhe/IR/OpenfheOps.td",
        False,
        ["@heir//lib/Dialect:td_files"],
    ),
    "orion": ("Orion/IR/OrionOps.td", False, []),
    "polynomial": ("Polynomial/IR/PolynomialOps.td", False, []),
    "poulpy": ("Poulpy/IR/PoulpyOps.td", True, []),
    "preprocessing": ("Preprocessing/IR/PreprocessingOps.td", False, []),
    "random": ("Random/IR/RandomOps.td", True, []),
    "rns": ("RNS/IR/RNSOps.td", False, []),
    "scifrbool": (
        "SCIFRBool/IR/SCIFRBoolOps.td",
        False,
        [
            "@heir//lib/Dialect/LWE/IR:td_files",
            "@heir//lib/Dialect/Polynomial/IR:td_files",
            "@heir//lib/Dialect:td_files",
        ],
    ),
    "secret": ("Secret/IR/SecretOps.td", False, []),
    "tensor_ext": ("TensorExt/IR/TensorExtOps.td", False, []),
    "tfhe_rust": ("TfheRust/IR/TfheRustOps.td", False, []),
    "tfhe_rust_bool": ("TfheRustBool/IR/TfheRustBoolOps.td", False, []),
}

[
    heir_py_dialect(
        name = "%s_py" % dialect,
        dialect = dialect,
        enums = enums,
        td_file = "@heir//lib/Dialect/%s" % td.replace("/IR/", "/IR:"),
        deps = ["@heir//lib/Dialect/%s:td_files" % td.rsplit("/", 1)[0]] + deps,
    )
    for dialect, (td, enums, deps) in HEIR_DIALECTS.items()
]

mlir_python_package(
    name = "heir_pkg",
    package_name = "heir",
    srcs = glob(["heir/**/*.py"]) + [
        ":%s_py" % dialect
        for dialect in HEIR_DIALECTS
    ],
    native_libs = [
        "@heir//lib/Bindings/Python:_heir.so",
        "@heir//lib/Bindings/Python:_mlir.so",
        "@heir//lib/Bindings/Python:libHEIRPythonCAPI.so",
    ],
    upstream_srcs = [
        "@llvm-project//mlir/python:AffineOpsPyFiles",
        "@llvm-project//mlir/python:ArithOpsPyFiles",
        "@llvm-project//mlir/python:BufferizationOpsPyFiles",
        "@llvm-project//mlir/python:BuiltinOpsPyFiles",
        "@llvm-project//mlir/python:DialectCorePyFiles",
        "@llvm-project//mlir/python:EmitCPyFiles",
        "@llvm-project//mlir/python:ExtrasPyFiles",
        "@llvm-project//mlir/python:FuncPyFiles",
        "@llvm-project//mlir/python:IRPyFiles",
        "@llvm-project//mlir/python:MathOpsPyFiles",
        "@llvm-project//mlir/python:MemRefOpsPyFiles",
        "@llvm-project//mlir/python:MlirLibsPyFiles",
        "@llvm-project//mlir/python:PassManagerPyFiles",
        "@llvm-project//mlir/python:RewritePyFiles",
        "@llvm-project//mlir/python:SCFPyFiles",
        "@llvm-project//mlir/python:TensorOpsPyFiles",
    ],
)

py_library(
    name = "heir",
    data = [":heir_pkg"],
    imports = ["heir_pkg"],
)

py_test(
    name = "heir_test",
    srcs = ["test/heir_test.py"],
    deps = [":heir"],
)
