load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")

make(
    name = "yosys_make",
    args = [
        "-j16",
        "CONFIG=gcc",
        "ENABLE_TCL=0",
        "ENABLE_GLOB=0",
        "ENABLE_PLUGINS=0",
        "ENABLE_READLINE=0",
        "ENABLE_COVER=0",
        "ENABLE_ZLIB=0",
        "ENABLE_ABC=0",
        #"ENABLE_ABC=1",
        "YOSYS_DATDIR=1",
        # "ABCEXTERNAL=$(location @abc//:abc_bin)",
    ],
    # data = [
    #     "@abc//:abc_bin",
    # ],
    lib_source = ":yosys_srcs",
    out_binaries = ["yosys"],
    targets = [
        "yosys",
        "install",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "yosys",
    srcs = [":yosys_make"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "yosys_srcs",
    srcs = glob([
        "*",
        "**/*",
    ]),
)
