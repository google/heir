# MemrefToArith pass eliminates memrefs in favor of inline arithmetic constants.

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Utils",
    srcs = ["Utils.cpp"],
    hdrs = ["@heir//include/Conversion/MemrefToArith:Utils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "UtilsTest",
    srcs = ["UtilsTest.cpp"],
    deps = [
        ":Utils",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "MemrefGlobalReplace",
    srcs = ["MemrefGlobalReplace.cpp"],
    hdrs = ["@heir//include/Conversion/MemrefToArith:MemrefToArith.h"],
    deps = [
        ":Utils",
        "@heir//include/Conversion/MemrefToArith:pass_inc_gen",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ExpandCopy",
    srcs = ["ExpandCopy.cpp"],
    hdrs = ["@heir//include/Conversion/MemrefToArith:MemrefToArith.h"],
    deps = [
        "@heir//include/Conversion/MemrefToArith:pass_inc_gen",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ExtractLoopBody",
    srcs = ["ExtractLoopBody.cpp"],
    hdrs = ["@heir//include/Conversion/MemrefToArith:MemrefToArith.h"],
    deps = [
        "@heir//include/Conversion/MemrefToArith:pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "UnrollAndForward",
    srcs = ["UnrollAndForward.cpp"],
    hdrs = ["@heir//include/Conversion/MemrefToArith:MemrefToArith.h"],
    deps = [
        ":Utils",
        "@heir//include/Conversion/MemrefToArith:pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ArithDialect",
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
        ":ExtractLoopBody",
        ":MemrefGlobalReplace",
        ":UnrollAndForward",
        "@heir//include/Conversion/MemrefToArith:pass_inc_gen",
    ],
    alwayslink = 1,
)
