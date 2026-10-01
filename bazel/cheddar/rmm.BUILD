load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "rmm",
    hdrs = glob(
        [
            "include/rmm/**/*.hpp",
            "include/rmm/**/*.h",
            "include/rmm/**/*.cuh",
        ],
        allow_empty = True,
    ),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@cuda//:cuda_headers",
        "@cuda//:libcudacxx",
        "@cuda//:thrust",
        "@rules_cuda//cuda:runtime",
        "@spdlog",
    ],
)
