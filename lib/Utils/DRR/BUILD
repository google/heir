# Tablegen helpers for declarative rewrite (DRR) patterns
load("@llvm-project//mlir:tblgen.bzl", "td_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "DRR",
    srcs = [
        "Utils.td",
    ],
    # include from the heir-root to enable fully-qualified include-paths
    includes = ["../../.."],
)
