# HEIR, an MLIR project for homomorphic encryption

load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

license(name = "license")

licenses(["notice"])

exports_files([
    "LICENSE",
])

package_group(
    name = "internal",
    packages = [],
)

# Disables deps for CI tools and tests.
# use by passing `--//:enable_openmp=0` or `--//:enable_yosys=0`
# to `bazel build` or `bazel test`

# OpenMP
string_flag(
    name = "enable_openmp",
    # TODO(#1361): re-enable when it's compatible with the Python frontend
    build_setting_default = "0",
)

config_setting(
    name = "config_enable_openmp",
    flag_values = {
        ":enable_openmp": "1",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "config_disable_openmp",
    flag_values = {
        ":enable_openmp": "0",
    },
    visibility = ["//visibility:public"],
)

# Yosys
string_flag(
    name = "enable_yosys",
    build_setting_default = "1",
)

config_setting(
    name = "config_enable_yosys",
    flag_values = {
        ":enable_yosys": "1",
    },
)

config_setting(
    name = "config_disable_yosys",
    flag_values = {
        ":enable_yosys": "0",
    },
)
