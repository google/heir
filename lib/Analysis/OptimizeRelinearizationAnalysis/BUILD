package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OptimizeRelinearizationAnalysis",
    srcs = ["OptimizeRelinearizationAnalysis.cpp"],
    hdrs = ["OptimizeRelinearizationAnalysis.h"],
    deps = [
        "@com_google_absl//status:statusor",
        "@com_google_absl//time",
        "@com_google_ortools//math_opt/cpp:math_opt",
        "@com_google_ortools//math_opt/solvers:gscip_solver",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
