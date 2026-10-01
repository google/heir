"""Helpers for CHEDDAR's opt-in build configuration."""

def if_cheddar_enabled(if_true, if_false = []):
    """Selects a value based on whether CHEDDAR is enabled."""
    return select({
        "@heir//:config_enable_cheddar": if_true,
        "//conditions:default": if_false,
    })

def requires_cheddar():
    """Marks a target incompatible unless CHEDDAR is enabled on Linux x86."""
    return select({
        "@heir//:config_enable_cheddar_linux_x86_64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

def cheddar_deps(extra = []):
    """Returns CHEDDAR library dependencies when enabled."""
    return if_cheddar_enabled(["@cheddar//:cheddar"] + extra)
