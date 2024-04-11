package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ApplyFolders",
    srcs = ["ApplyFolders.cpp"],
    hdrs = [
        "@heir//include/Transforms/ApplyFolders:ApplyFolders.h",
    ],
    deps = [
        "@heir//include/Transforms/ApplyFolders:pass_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
