package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ForwardStoreToLoad",
    srcs = ["ForwardStoreToLoad.cpp"],
    hdrs = [
        "@heir//include/Transforms/ForwardStoreToLoad:ForwardStoreToLoad.h",
    ],
    deps = [
        "@heir//include/Transforms/ForwardStoreToLoad:pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
