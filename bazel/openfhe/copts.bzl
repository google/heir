"""Build settings for OpenFHE and OpenMP."""

_OPENMP_CLANG_LINKOPTS = [
    "-fopenmp",
    "-lomp",
]

_OPENMP_GCC_LINKOPTS = [
    "-fopenmp",
    "-lgomp",
]

OPENMP_LINKOPTS = select({
    "@openfhe//:clang_openmp": _OPENMP_CLANG_LINKOPTS,
    "@openfhe//:gcc_openmp": _OPENMP_GCC_LINKOPTS,
    "//conditions:default": [],
})

OPENMP_COPTS = select({
    "@openfhe//:config_enable_openmp": [
        "-fopenmp",
        "-Xpreprocessor",
        "-Wno-unused-command-line-argument",
    ],
    "//conditions:default": [],
})
