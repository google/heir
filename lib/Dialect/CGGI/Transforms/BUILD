package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "@heir//include/Dialect/CGGI/Transforms:Passes.h",
    ],
    deps = [
        ":SetDefaultParameters",
        ":StraightLineVectorizer",
        "@heir//include/Dialect/CGGI/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "SetDefaultParameters",
    srcs = ["SetDefaultParameters.cpp"],
    hdrs = [
        "@heir//include/Dialect/CGGI/Transforms:SetDefaultParameters.h",
    ],
    deps = [
        "@heir//include/Dialect/CGGI/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "StraightLineVectorizer",
    srcs = ["StraightLineVectorizer.cpp"],
    hdrs = [
        "@heir//include/Dialect/CGGI/Transforms:StraightLineVectorizer.h",
    ],
    deps = [
        "@heir//include/Dialect/CGGI/Transforms:pass_inc_gen",
        "@heir//include/Graph",
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
