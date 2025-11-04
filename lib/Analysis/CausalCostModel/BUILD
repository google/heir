package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CausalGraph",
    srcs = ["CausalGraph.cpp"],
    hdrs = ["CausalGraph.h"],
    deps = [],
)

cc_library(
    name = "CostModel",
    srcs = ["CostModel.cpp"],
    hdrs = ["CostModel.h"],
    deps = [
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "CausalCostModel",
    srcs = ["CausalCostModel.cpp"],
    hdrs = ["CausalCostModel.h"],
    deps = [
        ":CausalGraph",
        ":CostModel",
        "@heir//lib/Kernel:ArithmeticDag",
        "@heir//lib/Kernel:MultiplicativeDepthVisitor",
        "@heir//lib/Kernel:RotationCountVisitor",
        "@llvm-project//mlir:IR",
    ],
)
