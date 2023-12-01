package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "@heir//include/Dialect/Secret/Transforms:Passes.h",
    ],
    deps = [
        ":DistributeGeneric",
        ":ForgetSecrets",
        "@heir//include/Dialect/Secret/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "ForgetSecrets",
    srcs = ["ForgetSecrets.cpp"],
    hdrs = [
        "@heir//include/Dialect/Secret/Transforms:ForgetSecrets.h",
    ],
    deps = [
        "@heir//include/Dialect/Secret/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:SecretOps",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "DistributeGeneric",
    srcs = ["DistributeGeneric.cpp"],
    hdrs = [
        "@heir//include/Dialect/Secret/Transforms:DistributeGeneric.h",
    ],
    deps = [
        "@heir//include/Dialect/Secret/Transforms:pass_inc_gen",
        "@heir//lib/Dialect/Secret/IR:SecretOps",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
