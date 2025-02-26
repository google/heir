package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Params",
    srcs = ["Params.cpp"],
    hdrs = ["Params.h"],
    deps = [
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Parameters:RLWEParams",
        "@llvm-project//llvm:Support",
    ],
)
