load("//tests/Examples/plaintext:test.bzl", "plaintext_test")

package(default_applicable_licenses = ["@heir//:license"])

plaintext_test(
    size = "enormous",
    src = "roberts_cross_64x64.mlir",
    data = [
        ":roberts_cross_mod_test",
        ":roberts_cross_test",
    ],
    tags = [
        "manual",
        "nofastbuild",
        "notap",
    ],
)

cc_library(
    name = "roberts_cross_test",
    srcs = [
        "memrefCopy.cpp",
        "roberts_cross_test.c",
    ],
)

cc_library(
    name = "roberts_cross_mod_test",
    srcs = [
        "memrefCopy.cpp",
        "roberts_cross_mod_test.c",
    ],
)
