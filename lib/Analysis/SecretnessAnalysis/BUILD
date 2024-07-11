# SecretnessAnalysis analysis class
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SecretnessAnalysis",
    srcs = ["SecretnessAnalysis.cpp"],
    hdrs = ["SecretnessAnalysis.h"],
    deps = [
        "@heir//lib/Dialect:Utils",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
    ],
)
