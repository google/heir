load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "assign_layout_benchmark",
    srcs = ["assign_layout_benchmark.cpp"],
    deps = [
        "@google_benchmark//:benchmark_main",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Transforms/ConvertToCiphertextSemantics:AssignLayout",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
    ],
)
