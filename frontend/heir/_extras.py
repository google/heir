"""Helpers for gating optional features behind pip extras."""

import importlib.util


def require(dependency, extra):
  """Raise an ImportError if an optional dependency is missing.

  We check the absolute name of the third-party `dependency` the extra
  installs, rather than one of heir's own modules: extras gate pip
  dependencies, not which files ship in the wheel, so `find_spec("heir.foo")`
  is always non-None and would never detect a missing extra.

  Args:
    dependency: absolute name of a third-party module the extra installs (e.g.
      "numba").
    extra: name of the pip extra that provides it (e.g. "python").
  """
  if importlib.util.find_spec(dependency) is None:
    raise ImportError(
        f"Failed to import {dependency}. This suggests heir was installed with"
        f" pip and missing the {extra} extra; install it via `pip install"
        f" heir_py[{extra}]`."
    )
