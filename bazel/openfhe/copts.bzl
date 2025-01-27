"""Build settings for OpenFHE and OpenMP."""

OPENFHE_COPTS = [
    "-Wno-non-virtual-dtor",
    "-Wno-shift-op-parentheses",
    "-Wno-unused-private-field",
    "-fexceptions",
]

OPENFHE_LINKOPTS = [
    "-fopenmp",
    "-lomp",
]

OPENMP_COPTS = [
    "-fopenmp",
    "-Xpreprocessor",
    "-Wno-unused-command-line-argument",
]
