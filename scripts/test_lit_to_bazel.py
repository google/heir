from scripts.lit_to_bazel import (
    convert_to_run_commands,
    normalize_lit_test_file_arg,
    PIPE,
)


def test_convert_to_run_commands_simple():
  run_lines = [
      "// RUN: heir-opt --canonicalize",
  ]
  assert convert_to_run_commands(run_lines) == [
      "heir-opt --canonicalize",
  ]


def test_convert_to_run_commands_simple_with_filecheck():
  run_lines = [
      "// RUN: heir-opt --canonicalize | FileCheck %s",
  ]
  assert convert_to_run_commands(run_lines) == [
      "heir-opt --canonicalize",
      PIPE,
      "FileCheck %s",
  ]


def test_convert_to_run_commands_simple_with_line_continuation():
  run_lines = [
      "// RUN: heir-opt \\",
      "// RUN: --canonicalize | FileCheck %s",
  ]
  assert convert_to_run_commands(run_lines) == [
      "heir-opt --canonicalize",
      PIPE,
      "FileCheck %s",
  ]


def test_convert_to_run_commands_simple_with_multiple_line_continuations():
  run_lines = [
      "// RUN: heir-opt \\",
      "// RUN: --canonicalize \\",
      "// RUN: --cse | FileCheck %s",
  ]
  assert convert_to_run_commands(run_lines) == [
      "heir-opt --canonicalize --cse",
      PIPE,
      "FileCheck %s",
  ]


def test_convert_to_run_commands_simple_with_second_command():
  run_lines = [
      "// RUN: heir-opt --canonicalize > %t",
      "// RUN: FileCheck %s < %t",
  ]
  assert convert_to_run_commands(run_lines) == [
      "heir-opt --canonicalize > %t",
      "FileCheck %s < %t",
  ]


def test_convert_to_run_commands_simple_with_non_run_garbage():
  run_lines = [
      "// RUN: heir-opt --canonicalize > %t",
      "// wat",
      "// RUN: FileCheck %s < %t",
  ]
  assert convert_to_run_commands(run_lines) == [
      "heir-opt --canonicalize > %t",
      "FileCheck %s < %t",
  ]


def test_convert_to_run_commands_with_multiple_pipes():
  run_lines = [
      "// RUN: heir-opt --canonicalize \\",
      "// RUN: | heir-translate --emit-verilog \\",
      "// RUN: | FileCheck %s",
  ]
  assert convert_to_run_commands(run_lines) == [
      "heir-opt --canonicalize",
      PIPE,
      "heir-translate --emit-verilog",
      PIPE,
      "FileCheck %s",
  ]


def test_normalize_lit_test_file_arg():
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

  for input in inputs:
    actual = normalize_lit_test_file_arg(input)
    assert actual == expected
