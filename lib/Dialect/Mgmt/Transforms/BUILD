load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = ["Passes.h"],
    deps = [
        ":AnnotateMgmt",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "AnnotateMgmt",
    srcs = ["AnnotateMgmt.cpp"],
    hdrs = ["AnnotateMgmt.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Mgmt",
    td_file = "Passes.td",
)
