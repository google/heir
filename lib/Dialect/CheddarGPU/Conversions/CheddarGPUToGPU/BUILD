load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "CheddarGPUToGPU.h.inc": [
            "-gen-pass-decls",
            "-name=CheddarGPUToGPU",
        ],
        "CheddarGPUToGPU.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "CheddarGPUToGPU",
    srcs = [
        "BConvKernels.cpp",
        "CheddarGPUToGPU.cpp",
        "HoistKernels.cpp",
        "ModDownEpilogueKernels.cpp",
        "NTTFusedKernels.cpp",
        "NTTKernels.cpp",
    ],
    hdrs = [
        "BConvKernels.h",
        "CheddarGPUToGPU.h",
        "HoistKernels.h",
        "KernelBuilders.h",
        "ModDownEpilogueKernels.h",
        "NTTFusedKernels.h",
        "NTTKernels.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/CheddarGPU/IR:Dialect",
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Utils/CheddarRef:mont_arith",
        "@heir//lib/Utils/CheddarRef:params",
        "@heir//lib/Utils/CheddarRef:tables",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)
