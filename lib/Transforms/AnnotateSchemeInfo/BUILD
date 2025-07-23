load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "AnnotateSchemeInfo",
    srcs = ["AnnotateSchemeInfo.cpp"],
    hdrs = [
        "AnnotateSchemeInfo.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/SchemeInfoAnalysis",
        "@heir//lib/Analysis/SchemeSelectionAnalysis",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "AnnotateSchemeInfo",
)
