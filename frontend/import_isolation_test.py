"""Test that `import heir` does not import optional dependencies."""

import os
import subprocess
import sys
from absl.testing import absltest

OPTIONAL_DEPS = ("numba", "numpy", "pybind11")


class ImportIsolationTest(absltest.TestCase):

  def test_import_heir_does_not_load_optional_deps(self):
    # We execute the test in a clean subprocess to ensure it runs in process isolation
    # and is not polluted by other tests loaded by the test runner (like pytest).

    # Pass current sys.path via PYTHONPATH to the child process so it can find the modules.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    # The code to execute in the subprocess
    code = f"""
import importlib
import importlib.util
import sys

OPTIONAL_DEPS = {OPTIONAL_DEPS}

# Verify optional deps are installed
for dep in OPTIONAL_DEPS:
  if importlib.util.find_spec(dep) is None:
    print(f"Error: {{dep}} must be installed for this test to be meaningful", file=sys.stderr)
    sys.exit(2)

# Verify they are not preloaded in this clean process
preloaded = [m for m in (*OPTIONAL_DEPS, "heir") if m in sys.modules]
if preloaded:
  print(f"Error: {{preloaded}} preloaded before import", file=sys.stderr)
  sys.exit(3)

# Import heir
importlib.import_module("frontend.heir")

# Verify they were not loaded as side effects
leaked = [dep for dep in OPTIONAL_DEPS if dep in sys.modules]
if leaked:
  print(f"Error: `import heir` imported {{leaked}}", file=sys.stderr)
  sys.exit(4)

print("SUCCESS")
sys.exit(0)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
      print("Subprocess stdout:")
      print(result.stdout)
      print("Subprocess stderr:")
      print(result.stderr)

    self.assertEqual(
        result.returncode,
        0,
        f"Import isolation check failed with exit code {result.returncode}. See"
        " stdout/stderr for details.",
    )


if __name__ == "__main__":
  absltest.main()
