# OpenFhePke Emitter

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OpenFhePkeEmitter",
    srcs = ["OpenFhePkeEmitter.cpp"],
    hdrs = [
        "OpenFhePkeTemplates.h",
        "@heir//include/Target/OpenFhePke:OpenFhePkeEmitter.h",
    ],
    deps = [
        "@heir//lib/Analysis/SelectVariableNames",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Target:Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TranslateLib",
    ],
)
