# MemrefToArith pass eliminates memrefs in favor of inline arithmetic constants.

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Utils",
    srcs = ["Utils.cpp"],
    hdrs = ["@heir//include/Conversion/MemrefToArith:Utils.h"],
    includes = ["@heir//include"],
    deps = [
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
    ],
)

cc_test(
    name = "UtilsTest",
    srcs = ["UtilsTest.cpp"],
    includes = ["@heir//include"],
    deps = [
        ":Utils",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
    ],
)

cc_library(
    name = "MemrefGlobalReplace",
    srcs = ["MemrefGlobalReplace.cpp"],
    hdrs = ["@heir//include/Conversion/MemrefToArith:MemrefToArith.h"],
    includes = ["@heir//include"],
    deps = [
        ":Utils",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ExpandCopy",
    srcs = ["ExpandCopy.cpp"],
    hdrs = ["@heir//include/Conversion/MemrefToArith:MemrefToArith.h"],
    includes = ["@heir//include"],
    deps = [
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "UnrollAndForward",
    srcs = ["UnrollAndForward.cpp"],
    hdrs = ["@heir//include/Conversion/MemrefToArith:MemrefToArith.h"],
    includes = ["@heir//include"],
    deps = [
        ":Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "MemrefToArithRegistration",
    srcs = ["PassRegistration.cpp"],
    deps = [
        ":ExpandCopy",
        ":MemrefGlobalReplace",
        ":UnrollAndForward",
        "@heir//include/Conversion/MemrefToArith:pass_inc_gen",
    ],
    alwayslink = 1,
)
