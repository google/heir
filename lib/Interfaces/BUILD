# HEIR project-wide interface implementations
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "NoiseInterfaces",
    srcs = ["NoiseInterfaces.cpp"],
    hdrs = ["@heir//include/Interfaces:NoiseInterfaces.h"],
    deps = [
        "@heir//include/Interfaces:NoiseInterfacesIncGen",
        "@heir//lib/Analysis/NoisePropagation:Variance",
        "@llvm-project//mlir:IR",
    ],
)
