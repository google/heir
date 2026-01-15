load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "BooleanVectorizer",
    srcs = ["BooleanVectorizer.cpp"],
    hdrs = [
        "BooleanVectorizer.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:HEIRInterfaces",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Utils/Graph",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "BooleanVectorizer",
)
