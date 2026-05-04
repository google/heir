"""Binary entry point for e2e_to_bazel."""

import fire
from scripts import e2e_to_bazel_lib

if __name__ == "__main__":
  fire.Fire(e2e_to_bazel_lib.e2e_to_bazel)
