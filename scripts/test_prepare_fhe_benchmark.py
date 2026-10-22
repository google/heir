"""Tests for packaging HEIR artifacts with the MNIST benchmark scaffold."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import prepare_fhe_benchmark


class PrepareSubmissionTest(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp_dir.cleanup)
    self.root = Path(self.temp_dir.name)
    self.reference = self.root / "reference"
    # Use distinct bytes for every file to detect accidental replacement.
    for name in prepare_fhe_benchmark.REFERENCE_FILES:
      path = self.reference / name
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_bytes(f"reference: {name}\n".encode())
    self.artifacts = {}
    for name in ("source", "header", "mlir", "weights"):
      path = self.root / name
      path.write_bytes(b"\x00\xff" + name.encode())
      self.artifacts[name] = path
    self.output = self.root / "submission"

  def prepare(self):
    prepare_fhe_benchmark.prepare_submission(
        self.reference, self.output, **self.artifacts
    )

  def test_packages_only_required_files_and_preserves_inputs(self):
    stale = self.reference / "build/stale.o"
    stale.parent.mkdir()
    stale.write_bytes(b"do not copy")
    old_model = self.reference / "data/traced_model.pt"
    old_model.parent.mkdir()
    old_model.write_bytes(b"old weights")
    old_docs = self.reference / "docs/README.md"
    old_docs.parent.mkdir()
    old_docs.write_text("old accuracy claims", encoding="utf-8")

    self.prepare()

    expected_artifacts = {
        "src/mlp_openfhe.cpp": "source",
        "include/mlp_openfhe.h": "header",
        "docs/mlp.mlir": "mlir",
        "data/traced_model.pt": "weights",
    }
    for destination, artifact in expected_artifacts.items():
      self.assertEqual(
          (self.output / destination).read_bytes(),
          self.artifacts[artifact].read_bytes(),
      )
    for name in prepare_fhe_benchmark.REFERENCE_FILES:
      self.assertEqual(
          (self.output / name).read_bytes(),
          f"reference: {name}\n".encode(),
      )
      self.assertEqual(
          (self.reference / name).read_bytes(),
          (self.output / name).read_bytes(),
      )
    expected = set(prepare_fhe_benchmark.REFERENCE_FILES) | set(
        expected_artifacts
    )
    expected.update(("README.md", "docs/README.md"))
    self.assertEqual(
        {
            p.relative_to(self.output).as_posix()
            for p in self.output.rglob("*")
            if p.is_file()
        },
        expected,
    )
    self.assertEqual(old_model.read_bytes(), b"old weights")
    self.assertEqual(
        old_docs.read_text(encoding="utf-8"), "old accuracy claims"
    )
    self.assertIn("TODO", (self.output / "docs/README.md").read_text())
    self.assertIn("ciphertext layout", (self.output / "README.md").read_text())

  def test_refuses_existing_output_without_changing_it(self):
    self.output.mkdir()
    sentinel = self.output / "keep.txt"
    sentinel.write_bytes(b"keep")
    with self.assertRaises(FileExistsError):
      self.prepare()
    self.assertEqual(list(self.output.iterdir()), [sentinel])
    self.assertEqual(sentinel.read_bytes(), b"keep")

  def test_missing_reference_file_leaves_no_output(self):
    (self.reference / "src/server_encrypted_compute.cpp").unlink()
    with self.assertRaisesRegex(FileNotFoundError, "server_encrypted_compute"):
      self.prepare()
    self.assertFalse(self.output.exists())

  def test_missing_artifact_leaves_no_output(self):
    self.artifacts["weights"].unlink()
    with self.assertRaisesRegex(FileNotFoundError, "weights"):
      self.prepare()
    self.assertFalse(self.output.exists())

  def test_refuses_output_inside_reference(self):
    self.output = self.reference / "nested"
    with self.assertRaisesRegex(ValueError, "outside the reference"):
      self.prepare()
    self.assertFalse(self.output.exists())

  def test_cli_reports_missing_input_without_traceback(self):
    command = [sys.executable, str(Path(prepare_fhe_benchmark.__file__))]
    arguments = {"reference": self.reference, "output": self.output}
    arguments.update(self.artifacts)
    self.artifacts["header"].unlink()
    for name, path in arguments.items():
      command.extend((f"--{name}", str(path)))
    result = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    self.assertEqual(result.returncode, 2)
    self.assertIn("Required file not found", result.stderr)
    self.assertNotIn("Traceback", result.stderr)
    self.assertFalse(self.output.exists())


if __name__ == "__main__":
  unittest.main()
