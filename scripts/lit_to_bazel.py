"""Binary entry point for lit_to_bazel."""

import fire
from scripts import lit_to_bazel_lib

if __name__ == "__main__":
  fire.Fire(lit_to_bazel_lib.lit_to_bazel)
