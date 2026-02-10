load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OptimizeRelinearizationAnalysis",
    srcs = ["OptimizeRelinearizationAnalysis.cpp"],
    hdrs = ["OptimizeRelinearizationAnalysis.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_ortools//ortools/math_opt/cpp:math_opt",
        "@com_google_ortools//ortools/math_opt/solvers:gscip_solver",
        "@heir//lib/Analysis/DimensionAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
