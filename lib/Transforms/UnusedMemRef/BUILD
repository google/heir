package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "UnusedMemRef",
    srcs = ["UnusedMemRef.cpp"],
    hdrs = [
        "@heir//include/Transforms/UnusedMemRef:UnusedMemRef.h",
    ],
    deps = [
        "@heir//include/Transforms/UnusedMemRef:pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
