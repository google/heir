# BUILD file for a bazel-native OpenFHE build
load("@heir//bazel/openfhe:copts.bzl", "OPENFHE_COPTS", "OPENFHE_LINKOPTS", "OPENMP_COPTS")

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-layering_check",  # Incompatible with `#include "gtest/gtest.h"`
        "-use_header_modules",  # Incompatible with -fexceptions.
    ],
)

licenses(["notice"])

OPENFHE_VERSION_MAJOR = 1

OPENFHE_VERSION_MINOR = 11

OPENFHE_VERSION_PATCH = 3

OPENFHE_VERSION = "{}.{}.{}".format(OPENFHE_VERSION_MAJOR, OPENFHE_VERSION_MINOR, OPENFHE_VERSION_PATCH)

OPENFHE_DEFINES = [
    "MATHBACKEND=2",
    "OPENFHE_VERSION=" + OPENFHE_VERSION,
    "PARALLEL",
]

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
    copts = OPENFHE_COPTS + OPENMP_COPTS + [
        # /utils/blockAllocator/blockAllocator.cpp has misaligned-pointer-use
        "-fno-sanitize=alignment",
    ],
    defines = OPENFHE_DEFINES,
    includes = [
        "src/core/include",
        "src/core/lib",
    ],
    linkopts = OPENFHE_LINKOPTS,
    textual_hdrs = glob([
        "src/core/include/**/*.h",
        "src/core/lib/**/*.cpp",
    ]),
    deps = ["@cereal"],
)

cc_library(
    name = "binfhe",
    srcs = glob([
        "src/binfhe/lib/**/*.c",
        "src/binfhe/lib/**/*.cpp",
    ]),
    copts = OPENFHE_COPTS + OPENMP_COPTS,
    defines = OPENFHE_DEFINES,
    includes = [
        "src/binfhe/include",
        "src/binfhe/lib",
    ],
    linkopts = OPENFHE_LINKOPTS,
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
    copts = OPENFHE_COPTS + OPENMP_COPTS + [
        "-Wno-vla-extension",
    ],
    defines = OPENFHE_DEFINES,
    includes = [
        "src/pke/include",
        "src/pke/lib",
    ],
    linkopts = OPENFHE_LINKOPTS,
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
