load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ApplyFolders",
    srcs = ["ApplyFolders.cpp"],
    hdrs = [
        "ApplyFolders.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    header_filename = "ApplyFolders.h.inc",
    pass_name = "ApplyFolders",
    td_file = "ApplyFolders.td",
)
