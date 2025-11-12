"""Build settings for OpenFHE and OpenMP."""

OPENFHE_VERSION_MAJOR = 1

OPENFHE_VERSION_MINOR = 11

OPENFHE_VERSION_PATCH = 3

OPENFHE_VERSION = "{}.{}.{}".format(OPENFHE_VERSION_MAJOR, OPENFHE_VERSION_MINOR, OPENFHE_VERSION_PATCH)

OPENFHE_DEFINES = [
    "MATHBACKEND=2",
    "OPENFHE_VERSION=" + OPENFHE_VERSION,
] + select({
    "@heir//:config_enable_openmp": ["PARALLEL"],
    "@heir//:config_disable_openmp": [],
})

OPENFHE_COPTS = [
    "-Wno-non-virtual-dtor",
    "-Wno-shift-op-parentheses",
    "-Wno-unused-private-field",
    # limit glibc and glibcxx to manylinux_2_28
    "-fno-exceptions",
    "-D_GLIBC_USE_DEPRECATED_SCANF",
    "-D_GNU_SOURCE=0",
]

_OPENFHE_LINKOPTS = [
    "-fopenmp",
    "-lomp",
]

_OPENMP_COPTS = [
    "-fopenmp",
    "-Xpreprocessor",
    "-Wno-unused-command-line-argument",
]

MAYBE_OPENFHE_LINKOPTS = select({
    "@heir//:config_enable_openmp": _OPENFHE_LINKOPTS,
    "@heir//:config_disable_openmp": [],
})

MAYBE_OPENMP_COPTS = select({
    "@heir//:config_enable_openmp": _OPENMP_COPTS,
    "@heir//:config_disable_openmp": [],
})
