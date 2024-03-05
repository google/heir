package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "FullLoopUnroll",
    srcs = ["FullLoopUnroll.cpp"],
    hdrs = [
        "@heir//include/Transforms/FullLoopUnroll:FullLoopUnroll.h",
    ],
    deps = [
        "@heir//include/Transforms/FullLoopUnroll:pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
