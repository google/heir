load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ConstQualifierAnalysis",
    srcs = ["ConstQualifierAnalysis.cpp"],
    hdrs = ["ConstQualifierAnalysis.h"],
    deps = [
        "@heir//lib/Utils/Tablegen:InPlaceOpInterface",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)
