# Metadata Emitter
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = ["MetadataEmitter.h"],
)

cc_library(
    name = "metadata_emitter",
    hdrs = ["MetadataEmitter.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
    ],
)
