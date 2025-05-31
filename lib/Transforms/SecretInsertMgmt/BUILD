load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "SecretInsertMgmt",
    srcs = [
        "SecretInsertMgmtBFV.cpp",
        "SecretInsertMgmtBGV.cpp",
        "SecretInsertMgmtCKKS.cpp",
        "SecretInsertMgmtPlaintext.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":SecretInsertMgmtPatterns",
        ":pass_inc_gen",
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/MulDepthAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect:ModuleAttributes",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/Transforms",
        "@heir//lib/Dialect/Mgmt/Transforms:AnnotateMgmt",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "SecretInsertMgmtPatterns",
    srcs = [
        "SecretInsertMgmtPatterns.cpp",
    ],
    hdrs = [
        "SecretInsertMgmtPatterns.h",
    ],
    deps = [
        "@heir//lib/Analysis/LevelAnalysis",
        "@heir//lib/Analysis/MulDepthAnalysis",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

add_heir_transforms(
    header_filename = "Passes.h.inc",
    pass_name = "SecretInsertMgmt",
    td_file = "Passes.td",
)
