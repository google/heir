# RotationAnalysis analysis pass
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RotationAnalysis",
    srcs = ["RotationAnalysis.cpp"],
    hdrs = ["@heir//include/Analysis/RotationAnalysis:RotationAnalysis.h"],
    deps = [
        "@heir//lib/Dialect:Utils",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TensorDialect",
    ],
)
