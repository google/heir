"""Tests for lit_to_bazel.py."""

from absl.testing import absltest
from scripts import lit_to_bazel_lib

patch = absltest.mock.patch

PIPE = lit_to_bazel_lib.PIPE
convert_to_run_commands = lit_to_bazel_lib.convert_to_run_commands
normalize_lit_test_file_arg = lit_to_bazel_lib.normalize_lit_test_file_arg


class LitToBazelTest(absltest.TestCase):
  """Tests for lit_to_bazel script."""

  def test_convert_to_run_commands_simple(self):
    run_lines = [
        "// RUN: heir-opt --canonicalize",
    ]
    self.assertEqual(
        convert_to_run_commands(run_lines),
        [
            "heir-opt --canonicalize",
        ],
    )

  def test_convert_to_run_commands_simple_with_filecheck(self):
    run_lines = [
        "// RUN: heir-opt --canonicalize | FileCheck %s",
    ]
    self.assertEqual(
        convert_to_run_commands(run_lines),
        [
            "heir-opt --canonicalize",
            PIPE,
            "FileCheck %s",
        ],
    )

  def test_convert_to_run_commands_simple_with_line_continuation(self):
    run_lines = [
        "// RUN: heir-opt \\",
        "// RUN: --canonicalize | FileCheck %s",
    ]
    self.assertEqual(
        convert_to_run_commands(run_lines),
        [
            "heir-opt --canonicalize",
            PIPE,
            "FileCheck %s",
        ],
    )

  def test_convert_to_run_commands_simple_with_multiple_line_continuations(
      self,
  ):
    run_lines = [
        "// RUN: heir-opt \\",
        "// RUN: --canonicalize \\",
        "// RUN: --cse | FileCheck %s",
    ]
    self.assertEqual(
        convert_to_run_commands(run_lines),
        [
            "heir-opt --canonicalize --cse",
            PIPE,
            "FileCheck %s",
        ],
    )

  def test_convert_to_run_commands_simple_with_second_command(self):
    run_lines = [
        "// RUN: heir-opt --canonicalize > %t",
        "// RUN: FileCheck %s < %t",
    ]
    self.assertEqual(
        convert_to_run_commands(run_lines),
        [
            "heir-opt --canonicalize > %t",
            "FileCheck %s < %t",
        ],
    )

  def test_convert_to_run_commands_simple_with_non_run_garbage(self):
    run_lines = [
        "// RUN: heir-opt --canonicalize > %t",
        "// wat",
        "// RUN: FileCheck %s < %t",
    ]
    self.assertEqual(
        convert_to_run_commands(run_lines),
        [
            "heir-opt --canonicalize > %t",
            "FileCheck %s < %t",
        ],
    )

  def test_convert_to_run_commands_with_multiple_pipes(self):
    run_lines = [
        "// RUN: heir-opt --canonicalize \\",
        "// RUN: | heir-translate --emit-verilog \\",
        "// RUN: | FileCheck %s",
    ]
    self.assertEqual(
        convert_to_run_commands(run_lines),
        [
            "heir-opt --canonicalize",
            PIPE,
            "heir-translate --emit-verilog",
            PIPE,
            "FileCheck %s",
        ],
    )

  def test_normalize_lit_test_file_arg(self):
    """Tests for normalize_lit_test_file_arg function."""
    inputs = [
        # no change
        "tests/Transforms/convert_to_ciphertext_semantics/tensor_extract.mlir",
        # bazel test target
        "//tests/Transforms/convert_to_ciphertext_semantics:tensor_extract.mlir.test",
        # No // prefix
        "tests/Transforms/convert_to_ciphertext_semantics:tensor_extract.mlir.test",
    ]
    expected = (
        "tests/Transforms/convert_to_ciphertext_semantics/tensor_extract.mlir"
    )

    for input_arg in inputs:
      actual = normalize_lit_test_file_arg(input_arg)
      self.assertEqual(actual, expected)

  def test_normalize_lit_test_file_arg_with_base_path(self):
    """Tests for normalize_lit_test_file_arg with base_path."""

    def side_effect(path):
      if path == "tests/foo.mlir":
        return False
      if path == "third_party/heir/tests/foo.mlir":
        return True
      return False

    with patch("os.path.exists", side_effect=side_effect):
      self.assertEqual(
          normalize_lit_test_file_arg(
              "tests/foo.mlir", base_path="third_party/heir"
          ),
          "third_party/heir/tests/foo.mlir",
      )


if __name__ == "__main__":
  absltest.main()
