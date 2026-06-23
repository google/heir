"""Build settings for OpenFHE and OpenMP."""

OPENMP_LINKOPTS = []

OPENMP_COPTS = select({
    "@openfhe//:config_enable_openmp": [
        "-fopenmp",
        "-Xpreprocessor",
        "-Wno-unused-command-line-argument",
    ],
    "//conditions:default": [],
})
