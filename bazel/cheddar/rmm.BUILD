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
        "@cuda//:cccl_headers",
        "@cuda//:cudart_headers",
        "@cuda//:thrust",
        "@rules_cuda//cuda:runtime",
        "@spdlog",
    ],
)
