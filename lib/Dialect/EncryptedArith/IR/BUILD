# EncryptedArith dialect

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "EncryptedArithDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/EncryptedArith/IR:EncryptedArithDialect.h",
        "@heir//include/Dialect/EncryptedArith/IR:EncryptedArithTypes.h",
    ],
    deps = [
        "@heir//include/Dialect/EncryptedArith/IR:dialect_inc_gen",
        "@heir//include/Dialect/EncryptedArith/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
