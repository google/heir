# Metadata Emitter

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "MetadataEmitter",
    srcs = ["MetadataEmitter.cpp"],
    hdrs = ["@heir//include/Target/Metadata:MetadataEmitter.h"],
    deps = [
        "@heir//include/Target/Metadata:metadata_emitter",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
