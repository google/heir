# MemrefToArith pass eliminates memrefs in favor of inline arithmetic constants.

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = [
        "@heir//:internal",
    ],
)

cc_library(
    name = "MemrefToArith",
    srcs = ["MemrefToArith.cpp"],
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
    name = "MemrefToArithRegistration",
    srcs = ["PassRegistration.cpp"],
    deps = [
        ":MemrefToArith",
        "@heir//include/Conversion/MemrefToArith:pass_inc_gen",
    ],
    alwayslink = 1,
)
