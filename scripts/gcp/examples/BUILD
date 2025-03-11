# GCP TPU examples for HEIR

load("@heir//tools:heir-jaxite.bzl", "fhe_jaxite_lib")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

fhe_jaxite_lib(
    name = "add_one_lut3",
    heir_opt_pass_flags = ["--tosa-to-boolean-jaxite"],
    mlir_src = "add_one_lut3.mlir",
    py_lib_target_name = "add_one_lut3_py_lib",
)
