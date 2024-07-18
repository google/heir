# AutoHoG importer library

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "AutoHogImporter",
    srcs = ["AutoHogImporter.cpp"],
    hdrs = ["AutoHogImporter.h"],
    deps = [
        "@heir//lib/Dialect/CGGI/IR:Dialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Graph",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TranslateLib",
        "@rapidjson",
    ],
)
