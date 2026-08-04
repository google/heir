load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "CompilationTarget",
    srcs = ["CompilationTarget.cpp"],
    hdrs = ["CompilationTarget.h"],
    deps = [
        ":compilation_target_overrides_inc_gen",
        ":compilation_target_struct_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

td_library(
    name = "td_files",
    srcs = ["HEIRTarget.td"],
    includes = ["../../../.."],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

td_library(
    name = "all_targets_td_files",
    srcs = ["AllTargets.td"],
    includes = ["../../.."],
    deps = [
        ":td_files",
        "@heir//lib/Target/Lattigo:lattigo_backend_config_td",
        "@heir//lib/Target/OpenFhePke:openfhe_backend_config_td",
    ],
)

gentbl_cc_library(
    name = "compilation_target_overrides_inc_gen",
    tbl_outs = {
        "CompilationTargetOverrides.cpp.inc": ["-gen-compilation-target-overrides"],
    },
    tblgen = "@heir//lib/Tablegen:heir-tblgen",
    td_file = "HEIRTarget.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "compilation_target_struct_inc_gen",
    tbl_outs = {
        "CompilationTargetStruct.h.inc": ["-gen-compilation-target-struct"],
    },
    tblgen = "@heir//lib/Tablegen:heir-tblgen",
    td_file = "AllTargets.td",
    deps = [
        ":all_targets_td_files",
    ],
)

cc_library(
    name = "RegisterAllBackends",
    srcs = ["RegisterAllBackends.cpp"],
    hdrs = ["RegisterAllBackends.h"],
    deps = [
        ":CompilationTarget",
        "@heir//lib/Target/Lattigo:lattigo_backend_config",
        "@heir//lib/Target/OpenFhePke:openfhe_backend_config",
    ],
)

bzl_library(
    name = "defs_bzl",
    srcs = ["defs.bzl"],
    visibility = ["//visibility:public"],
)

exports_files(["HEIRTarget.td"])
