load("@llvm-project//mlir:build_defs.bzl", "mlir_c_api_cc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Native code for HEIR's nanobind-based MLIR python bindings. The python
# package itself is assembled in //python. The layout follows upstream MLIR's
# CMake build (and JAX's bazel build):
#
#   heir/
#     heir-opt, heir-translate   thin launchers that run the tools from
#                                libHEIRPythonCAPI.so
#     _mlir_libs/
#       libHEIRPythonCAPI.so     ALL compiler code (MLIR + HEIR), exactly one copy
#       _mlir.so                 upstream's core extension (ir, passmanager, ...)
#       _heir.so                 HEIR registration + heir-translate access
#
# Both extensions link libHEIRPythonCAPI.so dynamically, so that there is only
# one copy of MLIR's global state (TypeIDs, pass registry, ...) in the process.
# Never link MLIR statically into an extension module.
#
# libHEIRPythonCAPI.so only exports the MLIR and HEIR C APIs, and the extensions
# only export their module init functions (see the *.lds and *.exp files).

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

config_setting(
    name = "opt_linux",
    constraint_values = ["@platforms//os:linux"],
    values = {"compilation_mode": "opt"},
)

config_setting(
    name = "opt_macos",
    constraint_values = ["@platforms//os:macos"],
    values = {"compilation_mode": "opt"},
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

# Optimized builds (like the wheel's) also drop the static symbol tables, which
# only debuggers and profilers use. (`--strip=always` only drops debug info.)
STRIP_LINKOPTS = select({
    ":opt_linux": ["-Wl,--strip-all"],
    ":opt_macos": ["-Wl,-x"],
    "//conditions:default": [],
})

# Only export the module init functions from the extension modules.
EXTENSION_LINKER_INPUTS = [
    "python_extension.exp",
    "python_extension.lds",
]

EXTENSION_LINKOPTS = PY_LINKOPTS + STRIP_LINKOPTS + select({
    "@platforms//os:macos": ["-Wl,-exported_symbols_list,$(execpath python_extension.exp)"],
    "//conditions:default": ["-Wl,--version-script=$(execpath python_extension.lds)"],
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
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_binary(
    name = "libHEIRPythonCAPI.so",
    additional_linker_inputs = [
        "libHEIRPythonCAPI.exp",
        "libHEIRPythonCAPI.lds",
    ],
    linkopts = select({
        "@platforms//os:macos": [
            "-Wl,-install_name,@rpath/libHEIRPythonCAPI.so",
            "-Wl,-exported_symbols_list,$(execpath libHEIRPythonCAPI.exp)",
        ],
        "//conditions:default": [
            "-Wl,-soname,libHEIRPythonCAPI.so",
            "-Wl,--version-script=$(execpath libHEIRPythonCAPI.lds)",
        ],
    }) + PY_LINKOPTS + STRIP_LINKOPTS,
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
    additional_linker_inputs = EXTENSION_LINKER_INPUTS,
    copts = PY_COPTS,
    features = PY_FEATURES,
    linkopts = EXTENSION_LINKOPTS,
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
    additional_linker_inputs = EXTENSION_LINKER_INPUTS,
    copts = PY_COPTS,
    features = PY_FEATURES,
    linkopts = EXTENSION_LINKOPTS,
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
# heir-opt and heir-translate for the wheel
##---------------------------------------------------------------------------##

# Thin launchers that run the tools from libHEIRPythonCAPI.so, which they
# expect in _mlir_libs/ next to them (see //python:heir_pkg), instead of linking
# another copy of the compiler. The standalone tools are //tools:heir-opt and
# //tools:heir-translate.
LAUNCHER_LINKOPTS = select({
    "@platforms//os:macos": ["-Wl,-rpath,@loader_path/_mlir_libs"],
    "//conditions:default": ["-Wl,-rpath,$$ORIGIN/_mlir_libs"],
}) + STRIP_LINKOPTS

cc_binary(
    name = "heir-opt-launcher",
    srcs = ["HeirOptLauncher.cpp"],
    linkopts = LAUNCHER_LINKOPTS,
    deps = [":HEIRPythonCAPIShared"],
)

cc_binary(
    name = "heir-translate-launcher",
    srcs = ["HeirTranslateLauncher.cpp"],
    linkopts = LAUNCHER_LINKOPTS,
    deps = [":HEIRPythonCAPIShared"],
)
