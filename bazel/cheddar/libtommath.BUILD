load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "tommath",
    srcs = glob(
        ["*.c"],
        allow_empty = True,
        exclude = [
            "demo/**",
            "etc/**",
            "mtest/**",
        ],
    ),
    hdrs = glob(
        ["*.h"],
        allow_empty = True,
    ),
    includes = ["."],
    visibility = ["//visibility:public"],
)
