# Tests for HEIR
load("//bazel:lit.bzl", "glob_lit_tests")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":lit.cfg.py",
        "@heir//tools:heir-opt",
        "@heir//tools:heir-translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:count",
        "@llvm-project//llvm:lit",
        "@llvm-project//llvm:not",
        # copybara: run_lit.sh
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    # Note: the driver argument is silently ignored for the OSS version of this
    # macro, replacing it with a standard invocation of lit whose config is in
    # lit.cfg.py
    driver = "@heir//tests:run_lit.sh",
    exclude = [
        "Examples/common/*",
        # Has a custom driver
        "Examples/plaintext/*",
        # FPGA tests are not supported by blaze
        "Examples/tfhe_rust_bool/fpga/*",
        "Examples/tfhe_rust_hl/fpga/*",
    ],
    test_file_exts = ["mlir"],
)

exports_files(
    [
        "run_lit.sh",
    ],
)
