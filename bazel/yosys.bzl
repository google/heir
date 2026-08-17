"""Helpers for targets that require Yosys support."""

def requires_yosys():
    """Marks a target incompatible unless built with --//:enable_yosys=1.

    Incompatible targets are automatically skipped by wildcard patterns like
    `bazel build //...`.
    """
    return select({
        "@heir//:config_enable_yosys": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
