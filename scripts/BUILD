load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_binary(
    name = "generate_static_roots",
    srcs = ["generate_static_roots.py"],
    deps = ["//third_party/py/sympy"],
)
