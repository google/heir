load("@llvm-project//mlir:build_defs.bzl", "mlir_c_api_cc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Native code for HEIR's nanobind-based MLIR python bindings. The python
# package itself is assembled in //python. The layout follows upstream MLIR's
# CMake build (and JAX's bazel build):
#
#   heir/_mlir_libs/
#     libHEIRPythonCAPI.so   ALL compiler code (MLIR + HEIR), exactly one copy
#     _mlir.so               upstream's core extension (ir, passmanager, ...)
#     _heir.so               HEIR registration + heir-translate access
#
# Both extensions link libHEIRPythonCAPI.so dynamically, so that there is only
# one copy of MLIR's global state (TypeIDs, pass registry, ...) in the process.
# Never link MLIR statically into an extension module.

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Every extension module (and the recompiled upstream core sources) must agree
# on these, otherwise capsule names and nanobind type names won't match. The
# package prefix must match the python package name used in //python.
cc_library(
    name = "python_defines",
    defines = [
        "MLIR_PYTHON_PACKAGE_PREFIX=heir.",
        "MLIR_BINDINGS_PYTHON_DOMAIN=heir",
        "NB_DOMAIN=heir",
    ],
)

PY_COPTS = [
    "-fexceptions",
    "-frtti",
]

PY_FEATURES = ["-use_header_modules"]

# Native extensions find libHEIRPythonCAPI.so next to themselves.
PY_LINKOPTS = select({
    "@platforms//os:macos": ["-Wl,-rpath,@loader_path"],
    "//conditions:default": ["-Wl,-rpath,$$ORIGIN"],
})

##---------------------------------------------------------------------------##
# C API (the only interface the python extensions use to talk to the compiler)
##---------------------------------------------------------------------------##

mlir_c_api_cc_library(
    name = "HEIRCAPI",
    srcs = ["HEIRCAPI.cpp"],
    hdrs = ["HEIRCAPI.h"],
    capi_deps = ["@llvm-project//mlir:CAPIIR"],
    deps = [
        "@heir//lib/Tools:HeirOptRegistration",
        "@heir//lib/Tools:HeirTranslateRegistration",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_binary(
    name = "libHEIRPythonCAPI.so",
    linkopts = select({
        "@platforms//os:macos": ["-Wl,-install_name,@rpath/libHEIRPythonCAPI.so"],
        "//conditions:default": ["-Wl,-soname,libHEIRPythonCAPI.so"],
    }) + PY_LINKOPTS,
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":HEIRCAPIObjects",
        "@llvm-project//mlir:MLIRBindingsPythonCAPIObjects",
    ],
)

cc_library(
    name = "HEIRPythonCAPIShared",
    srcs = [":libHEIRPythonCAPI.so"],
    deps = [
        ":HEIRCAPIHeaders",
        "@llvm-project//mlir:CAPIAsyncHeaders",
        "@llvm-project//mlir:CAPIDebugHeaders",
        "@llvm-project//mlir:CAPIIRHeaders",
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
    linkopts = PY_LINKOPTS,
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
    name = "_heir.so",
    srcs = ["HeirModule.cpp"],
    copts = PY_COPTS,
    features = PY_FEATURES,
    linkopts = PY_LINKOPTS,
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":HEIRPythonCAPIShared",
        ":python_defines",
        "@llvm-project//mlir:MLIRBindingsPythonNanobindHeaders",
        "@nanobind",
    ],
)
