"""Helper macros for CHEDDAR opt-in build configuration."""

def if_cheddar_enabled(if_true, if_false = []):
    """Select based on whether CHEDDAR is enabled."""
    return select({
        "@heir//:config_enable_cheddar": if_true,
        "//conditions:default": if_false,
    })

def requires_cheddar():
    """Returns target_compatible_with for CHEDDAR-requiring targets."""
    return select({
        "@heir//:config_enable_cheddar": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

def cheddar_deps(extra = []):
    """Returns CHEDDAR library deps, empty when disabled."""
    return if_cheddar_enabled(
        ["@cheddar//:cheddar"] + extra,
    )
