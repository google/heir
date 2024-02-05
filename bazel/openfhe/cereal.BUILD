package(
    default_visibility = ["//visibility:public"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
)

licenses(["notice"])

cc_library(
    name = "cereal",
    copts = ["-fexceptions"],
    includes = ["include"],
    textual_hdrs = glob([
        "include/**/*.hpp",
        "include/**/*.h",
    ]),
    deps = [
        "@rapidjson",
    ],
)
