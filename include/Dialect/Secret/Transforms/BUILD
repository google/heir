load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Secret",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "SecretPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

exports_files([
    "CaptureGenericAmbientScope.h",
    "GenericAbsorbConstants.h",
    "DistributeGeneric.h",
    "ForgetSecrets.h",
    "MergeAdjacentGenerics.h",
    "Passes.h",
])
