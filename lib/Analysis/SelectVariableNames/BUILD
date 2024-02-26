# SelectVariableNames analysis pass
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SelectVariableNames",
    srcs = ["SelectVariableNames.cpp"],
    hdrs = ["@heir//include/Analysis/SelectVariableNames:SelectVariableNames.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
