# Built in HEIR declarations

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "HEIRInterfaces",
    srcs = [
        "HEIRInterfaces.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect:HEIRInterfaces.h",
    ],
    deps = [
        "@heir//include/Dialect:interfaces_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "Utils",
    srcs = [
        "Utils.h",
    ],
    deps = [
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
