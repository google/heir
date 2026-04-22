load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "all_srcs",
    srcs = glob(
        ["**"],
        exclude = [
            "bazel-*/**",
            "build/**",
            "cmake-build*/**",
        ],
    ),
)

cmake(
    name = "cheddar_cmake",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_UNITTEST": "OFF",
        "ENABLE_EXTENSION": "ON",
        "USE_GMP": "OFF",
        # Build for the local GPU architecture to keep build times manageable.
        "CMAKE_CUDA_ARCHITECTURES": "native",
        # Assumes conventional CUDA toolkit location and configuration.
        "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
        "CUDAToolkit_ROOT": "/usr/local/cuda",
        "CMAKE_CUDA_HOST_COMPILER:FILEPATH": "/usr/bin/g++",
        "CMAKE_CUDA_FLAGS": "-ccbin=/usr/bin/g++",
        # Wipe toolchain-provided linker flags that can inject unsupported host options.
        "CMAKE_EXE_LINKER_FLAGS": "",
        "CMAKE_MODULE_LINKER_FLAGS": "",
        "CMAKE_SHARED_LINKER_FLAGS": "",
    },
    generate_crosstool_file = False,
    lib_source = ":all_srcs",
    out_include_dir = "include",
    out_shared_libs = [
        "libcheddar.so",
    ],
    targets = ["cheddar"],
)

# Wrap the cmake output with CUDA Thrust headers so downstream consumers
# can resolve cheddar's public #include <thrust/...> directives.
cc_library(
    name = "cheddar",
    deps = [
        ":cheddar_cmake",
        "@cuda//:thrust",
    ],
)
