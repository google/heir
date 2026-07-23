load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SoftmaxToCgfSoftmax",
    srcs = ["SoftmaxToCgfSoftmax.cpp"],
    hdrs = ["SoftmaxToCgfSoftmax.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/MathExt/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "SoftmaxToCgfSoftmax",
    td_file = "SoftmaxToCgfSoftmax.td",
)

cc_binary(
    name = "cgf_softmax_fuzz",
    srcs = ["cgf_softmax_fuzz.cpp"],
    deps = [
        "@com_google_absl//absl/random",
    ],
)
