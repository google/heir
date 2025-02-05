# BUILD file for a bazel-native pocketfft build
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "pocketfft",
    hdrs = [
        "pocketfft_hdronly.h",
    ],
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
)
