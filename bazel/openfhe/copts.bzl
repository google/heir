"""Build settings for OpenFHE and OpenMP."""

OPENMP_LINKOPTS = []

OPENMP_COPTS = [
    "-fopenmp",
    "-Xpreprocessor",
    "-Wno-unused-command-line-argument",
]
