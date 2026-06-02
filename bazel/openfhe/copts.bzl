"""Build settings for OpenFHE and OpenMP."""

# Under HEIR's hermetic, bootstrapped LLVM toolchain there is no system libomp
# to satisfy `-lomp`, and `-fopenmp` at link time would make the clang driver
# search for one. The OpenMP runtime is instead linked transitively via the
# @openfhe//:core / :pke deps, which now carry the hermetic @openmp//:libomp
# archive (see //patches:openfhe.patch). So the clang link line needs no extra
# OpenMP flags. (gcc, used only on non-hermetic hosts, still names -lgomp.)
_OPENMP_CLANG_LINKOPTS = []

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
