# A library build target that encapsulates the HEIR-generated code.
cc_library(
    name = "dot_product_codegen",
    srcs = ["heir_output.cpp"],
    hdrs = ["heir_output.h"],
    deps = ["@openfhe//:pke"],
)

# An executable build target that contains your main function and links
# against the above.
cc_binary(
    name = "dot_product_main",
    srcs = ["dot_product_main.cpp"],
    deps = [
        ":dot_product_codegen",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)
