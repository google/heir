# RNS dialect implementation

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Dialect",
    srcs = [
        "RNSDialect.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/RNS/IR:RNSDialect.h",
        "@heir//include/Dialect/RNS/IR:RNSOps.h",
        "@heir//include/Dialect/RNS/IR:RNSTypeInterfaces.h",
        "@heir//include/Dialect/RNS/IR:RNSTypes.h",
    ],
    deps = [
        ":RNSOps",
        ":RNSTypes",
        "@heir//include/Dialect/RNS/IR:dialect_inc_gen",
        "@heir//include/Dialect/RNS/IR:ops_inc_gen",
        "@heir//include/Dialect/RNS/IR:type_interfaces_inc_gen",
        "@heir//include/Dialect/RNS/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "RNSTypes",
    srcs = [
        "RNSTypes.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/RNS/IR:RNSDialect.h",
        "@heir//include/Dialect/RNS/IR:RNSTypeInterfaces.h",
        "@heir//include/Dialect/RNS/IR:RNSTypes.h",
    ],
    deps = [
        "@heir//include/Dialect/RNS/IR:dialect_inc_gen",
        "@heir//include/Dialect/RNS/IR:type_interfaces_inc_gen",
        "@heir//include/Dialect/RNS/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "RNSOps",
    srcs = [
        "RNSOps.cpp",
    ],
    hdrs = [
        "@heir//include/Dialect/RNS/IR:RNSDialect.h",
        "@heir//include/Dialect/RNS/IR:RNSOps.h",
        "@heir//include/Dialect/RNS/IR:RNSTypes.h",
    ],
    deps = [
        ":RNSTypes",
        "@heir//include/Dialect/RNS/IR:dialect_inc_gen",
        "@heir//include/Dialect/RNS/IR:ops_inc_gen",
        "@heir//include/Dialect/RNS/IR:types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
