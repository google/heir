load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "relations",
    hdrs = ["relations.h"],
)

cc_binary(
    name = "assign_layout_benchmark",
    srcs = ["assign_layout_benchmark.cpp"],
    deps = [
        ":relations",
        "@google_benchmark//:benchmark_main",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Transforms/ConvertToCiphertextSemantics:AssignLayout",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_binary(
    name = "card_estimation_benchmark",
    srcs = ["card_estimation_benchmark.cpp"],
    deps = [
        ":relations",
        "@google_benchmark//:benchmark_main",
        "@heir//lib/Utils/Layout:IslConversion",
        "@isl",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Support",
    ],
)

cc_binary(
    name = "enumerate_points_benchmark",
    srcs = ["enumerate_points_benchmark.cpp"],
    deps = [
        ":relations",
        "@google_benchmark//:benchmark_main",
        "@heir//lib/Utils/Layout:IslConversion",
        "@heir//lib/Utils/Layout:Utils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Support",
    ],
)

cc_binary(
    name = "get_ct_complement_benchmark",
    srcs = ["get_ct_complement_benchmark.cpp"],
    deps = [
        ":relations",
        "@google_benchmark//:benchmark_main",
        "@heir//lib/Utils/Layout:IslConversion",
        "@heir//lib/Utils/Layout:Utils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
