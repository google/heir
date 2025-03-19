load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

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
        ":AddDebugPort",
        ":CaptureGenericAmbientScope",
        ":DistributeGeneric",
        ":ExtractGenericBody",
        ":ForgetSecrets",
        ":GenericAbsorbConstants",
        ":GenericAbsorbDealloc",
        ":ImportExecutionResult",
        ":MergeAdjacentGenerics",
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:Dialect",
    ],
)

cc_library(
    name = "ForgetSecrets",
    srcs = ["ForgetSecrets.cpp"],
    hdrs = [
        "ForgetSecrets.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:SecretOps",
        "@heir//lib/Utils:ConversionUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "DistributeGeneric",
    srcs = ["DistributeGeneric.cpp"],
    hdrs = [
        "DistributeGeneric.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@heir//lib/Utils:AttributeUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "CaptureGenericAmbientScope",
    srcs = ["CaptureGenericAmbientScope.cpp"],
    hdrs = [
        "CaptureGenericAmbientScope.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "GenericAbsorbConstants",
    srcs = ["GenericAbsorbConstants.cpp"],
    hdrs = [
        "GenericAbsorbConstants.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "GenericAbsorbDealloc",
    srcs = ["GenericAbsorbDealloc.cpp"],
    hdrs = [
        "GenericAbsorbDealloc.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "MergeAdjacentGenerics",
    srcs = ["MergeAdjacentGenerics.cpp"],
    hdrs = [
        "MergeAdjacentGenerics.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "BufferizableOpInterface",
    srcs = [
        "BufferizableOpInterfaceImpl.cpp",
    ],
    hdrs = [
        "BufferizableOpInterfaceImpl.h",
    ],
    deps = [
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ExtractGenericBody",
    srcs = ["ExtractGenericBody.cpp"],
    hdrs = [
        "ExtractGenericBody.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:SecretPatterns",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "AddDebugPort",
    srcs = ["AddDebugPort.cpp"],
    hdrs = [
        "AddDebugPort.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect:FuncUtils",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ImportExecutionResult",
    srcs = ["ImportExecutionResult.cpp"],
    hdrs = [
        "ImportExecutionResult.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "Passes.h.inc": [
            "-gen-pass-decls",
            "-name=Secret",
        ],
        "SecretPasses.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
