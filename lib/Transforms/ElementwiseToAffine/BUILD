package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ElementwiseToAffine",
    srcs = ["ElementwiseToAffine.cpp"],
    hdrs = [
        "@heir//include/Transforms/ElementwiseToAffine:ElementwiseToAffine.h",
    ],
    deps = [
        "@heir//include/Transforms/ElementwiseToAffine:pass_inc_gen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
