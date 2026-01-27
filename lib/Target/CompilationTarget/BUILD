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
        ":compilation_target_inc_gen",
        "@heir//lib/Dialect:ModuleAttributes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
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
    name = "compilation_target_inc_gen",
    tbl_outs = [
        (
            ["-gen-compilation-target-registration"],
            "CompilationTarget.cpp.inc",
        ),
    ],
    tblgen = "@heir//lib/Tablegen:heir-tblgen",
    td_file = "HEIRTarget.td",
    deps = [
        ":td_files",
    ],
)

exports_files(["HEIRTarget.td"])
