"""Helper functions for pybind11.

We compile pybind11 bindings with clang, and from the
pybind11 docs, the main way to do this is

$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)

This file provides helper functions that implement the two shell subprocess steps

- pybind11_includes: $(python3 -m pybind11 --includes)
- pyconfig_ext_suffix: $(python3-config --extension-suffix)

"""
import sysconfig
from pybind11.__main__ import quote
from pybind11.commands import get_include


# Forked from pybind11 because the only implementation prints the values
# immediately.
def pybind11_includes() -> None:
    dirs = [
        sysconfig.get_path("include"),
        sysconfig.get_path("platinclude"),
        get_include(),
    ]

    # Make unique but preserve order
    unique_dirs = []
    for d in dirs:
        if d and d not in unique_dirs:
            unique_dirs.append(d)

    return [quote(d) for d in unique_dirs]


def pyconfig_ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX")
