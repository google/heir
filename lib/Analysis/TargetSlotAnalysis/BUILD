# TargetSlotAnalysis analysis pass
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "TargetSlotAnalysis",
    srcs = ["TargetSlotAnalysis.cpp"],
    hdrs = ["@heir//include/Analysis/TargetSlotAnalysis:TargetSlotAnalysis.h"],
    deps = [
        "@heir//lib/Dialect:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)
