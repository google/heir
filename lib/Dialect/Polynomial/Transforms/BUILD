load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":AttachNTTRoots",
        ":PolyMulToNTT",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
    ],
)

cc_library(
    name = "AttachNTTRoots",
    srcs = [
        "AttachNTTRoots.cpp",
    ],
    hdrs = [
        "AttachNTTRoots.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@heir//lib/Utils:APIntUtils",
        "@heir//lib/Utils:MathUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "PolyMulToNTT",
    srcs = [
        "NTTSolver.cpp",
        "PolyMulToNTT.cpp",
    ],
    hdrs = [
        "NTTSolver.h",
        "PolyMulToNTT.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@com_google_ortools//ortools/sat:cp_model",
        "@com_google_ortools//ortools/sat:cp_model_solver",
        "@heir//lib/Dialect/ModArith/IR:TypeInterfaces",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "Polynomial",
    td_file = "Passes.td",
)
