# MulDepthAnalysis analysis pass
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "MulDepthAnalysis",
    srcs = ["MulDepthAnalysis.cpp"],
    hdrs = ["MulDepthAnalysis.h"],
    deps = [
        "@heir//lib/Dialect:Utils",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
