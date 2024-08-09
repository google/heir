package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LazyRelinAnalysis",
    srcs = ["LazyRelinAnalysis.cpp"],
    hdrs = ["LazyRelinAnalysis.h"],
    deps = [
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@com_google_ortools//ortools/base",
        "@com_google_ortools//ortools/linear_solver",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
