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

gentbl_cc_library(
    name = "compilation_target_overrides_inc_gen",
    tbl_outs = [
        (
            ["-gen-compilation-target-overrides"],
            "CompilationTargetOverrides.cpp.inc",
        ),
    ],
    tblgen = "@heir//lib/Tablegen:heir-tblgen",
    td_file = "HEIRTarget.td",
    deps = [
        ":td_files",
    ],
)

gentbl_cc_library(
    name = "compilation_target_struct_inc_gen",
    tbl_outs = [
        (
            ["-gen-compilation-target-struct"],
            "CompilationTargetStruct.h.inc",
        ),
    ],
    tblgen = "@heir//lib/Tablegen:heir-tblgen",
    td_file = "HEIRTarget.td",
    deps = [
        ":td_files",
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
