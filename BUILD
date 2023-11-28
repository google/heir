# HEIR, an MLIR project for homomorphic encryption

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

# Disables Yosys deps for CLI tools and tests.
config_setting(
    name = "disable_yosys",
    values = {
        "define": "HEIR_NO_YOSYS=1",
    },
)
