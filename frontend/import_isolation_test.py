"""Test that `import heir` does not import optional dependencies."""

import importlib
import importlib.util
import sys

from absl.testing import absltest

OPTIONAL_DEPS = ("numba", "numpy", "pybind11")


class ImportIsolationTest(absltest.TestCase):

  def test_import_heir_does_not_load_optional_deps(self):
    # The deps must be installed (else this test is vacuous)
    # and not yet imported (else we couldn't attribute them to `import heir`).
    for dep in OPTIONAL_DEPS:
      self.assertIsNotNone(
          importlib.util.find_spec(dep),
          f"{dep} must be installed for this test to be meaningful",
      )
    preloaded = [m for m in (*OPTIONAL_DEPS, "heir") if m in sys.modules]
    self.assertEqual(preloaded, [], f"{preloaded} imported before the test ran")

    # This is normally `import heir` but this test is relative to the repository
    # root.
    importlib.import_module("frontend.heir")

    leaked = [dep for dep in OPTIONAL_DEPS if dep in sys.modules]
    self.assertEqual(
        leaked,
        [],
        f"Error: `import heir` imported {leaked}; but the base package should"
        " not import any of the optional dependencies",
    )


if __name__ == "__main__":
  absltest.main()
