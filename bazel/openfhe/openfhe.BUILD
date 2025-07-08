# BUILD file for a bazel-native OpenFHE build
load("@heir//bazel/openfhe:copts.bzl", "MAYBE_OPENFHE_LINKOPTS", "MAYBE_OPENMP_COPTS", "OPENFHE_COPTS", "OPENFHE_DEFINES")

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-layering_check",  # Incompatible with `#include "gtest/gtest.h"`
        "-use_header_modules",  # Incompatible with -fexceptions.
    ],
)

licenses(["notice"])

# This rule exists so that the python frontend can get access to the headers to
# pass dynamically to clang when building compiled code.
filegroup(
    name = "headers",
    srcs = glob([
        "src/binfhe/include/**/*.h",
        "src/core/include/**/*.h",
        "src/pke/include/**/*.h",
    ]),
)

cc_library(
    name = "core",
    srcs = glob([
        "src/core/lib/**/*.c",
        "src/core/lib/**/*.cpp",
    ]),
    copts = OPENFHE_COPTS + MAYBE_OPENMP_COPTS + [
        # /utils/blockAllocator/blockAllocator.cpp has misaligned-pointer-use
        "-fno-sanitize=alignment",
    ],
    defines = OPENFHE_DEFINES,
    includes = [
        "src/core/include",
        "src/core/lib",
    ],
    linkopts = MAYBE_OPENFHE_LINKOPTS,
    linkstatic = True,
    textual_hdrs = glob([
        "src/core/include/**/*.h",
        "src/core/lib/**/*.cpp",
    ]),
    deps = ["@cereal"],
)

cc_library(
    name = "binfhe",
    srcs = glob([
        "src/binfhe/lib/**/*.cpp",
    ]),
    copts = OPENFHE_COPTS + MAYBE_OPENMP_COPTS,
    defines = OPENFHE_DEFINES,
    includes = [
        "src/binfhe/include",
        "src/binfhe/lib",
    ],
    linkopts = MAYBE_OPENFHE_LINKOPTS,
    linkstatic = True,
    textual_hdrs = glob(["src/binfhe/include/**/*.h"]),
    deps = [
        "@openfhe//:core",
    ],
)

cc_library(
    name = "pke",
    srcs = glob([
        "src/pke/lib/**/*.cpp",
    ]),
    copts = OPENFHE_COPTS + MAYBE_OPENMP_COPTS + [
        "-Wno-vla-extension",
    ],
    defines = OPENFHE_DEFINES,
    includes = [
        "src/pke/include",
        "src/pke/lib",
    ],
    linkopts = MAYBE_OPENFHE_LINKOPTS,
    linkstatic = True,
    textual_hdrs = glob([
        "src/pke/include/**/*.h",
        "src/pke/lib/**/*.cpp",
    ]),
    deps = [
        "@cereal",
        "@openfhe//:binfhe",
        "@openfhe//:core",
    ],
)

# Explicitly needed because on some platforms bazel does not automatically
# generate a shared object file.
cc_shared_library(
    name = "core_shared",
    shared_lib_name = "libOPENFHEcore.so",
    deps = [":core"],
)

cc_shared_library(
    name = "binfhe_shared",
    shared_lib_name = "libOPENFHEbinfhe.so",
    deps = [":binfhe"],
)

cc_shared_library(
    name = "pke_shared",
    shared_lib_name = "libOPENFHEpke.so",
    deps = [":pke"],
)
