load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CompilationTargetEmitter",
    srcs = ["CompilationTargetEmitter.cpp"],
    hdrs = ["CompilationTargetEmitter.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
    ],
)

cc_binary(
    name = "heir-tblgen",
    srcs = ["TablegenMain.cpp"],
    deps = [
        ":CompilationTargetEmitter",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
    ],
)
