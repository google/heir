package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OptimizeRelinearizationAnalysis",
    srcs = ["OptimizeRelinearizationAnalysis.cpp"],
    hdrs = ["OptimizeRelinearizationAnalysis.h"],
    deps = [
        "@com_google_ortools//ortools/base",
        "@com_google_ortools//ortools/linear_solver",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)
