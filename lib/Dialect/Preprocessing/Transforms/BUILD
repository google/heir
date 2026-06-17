load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Preprocessing",
    td_file = "Passes.td",
)

cc_library(
    name = "Transforms",
    srcs = ["ValidatePreprocessing.cpp"],
    hdrs = [
        "Passes.h",
        "ValidatePreprocessing.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Preprocessing/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
