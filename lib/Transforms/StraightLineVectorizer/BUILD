package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "StraightLineVectorizer",
    srcs = ["StraightLineVectorizer.cpp"],
    hdrs = [
        "@heir//include/Transforms/StraightLineVectorizer:StraightLineVectorizer.h",
    ],
    deps = [
        "@heir//include/Graph",
        "@heir//include/Transforms/StraightLineVectorizer:pass_inc_gen",
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
