load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OptimizeRelinearization",
    srcs = ["OptimizeRelinearization.cpp"],
    hdrs = [
        "OptimizeRelinearization.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/OptimizeRelinearizationAnalysis",
        "@heir//lib/Dialect:Utils",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "OptimizeRelinearization",
)
