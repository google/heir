"""Unit tests for entrypoint.py."""

import io
import pathlib
import shutil
import unittest
from unittest.mock import Mock, patch

from scripts.torch_ci.entrypoint import CommandExecutor, CoverageRunner


class TestCoverageRunner(unittest.TestCase):

  def setUp(self):
    self.mock_executor = Mock(spec=CommandExecutor)
    self.models_dir = pathlib.Path("/tmp/mock_models")
    # Create a dummy directory structure for testing
    self.models_dir.mkdir(parents=True, exist_ok=True)
    (self.models_dir / "op1.py").touch()
    (self.models_dir / "op1.mlir").touch()

    self.runner = CoverageRunner(
        executor=self.mock_executor, models_dir=self.models_dir
    )

  def tearDown(self):
    # Clean up dummy directory
    if self.models_dir.exists():
      shutil.rmtree(self.models_dir)

  def test_find_models(self):
    models = self.runner.find_models()
    self.assertEqual(models, ["op1"])

  def test_run_export_success(self):
    self.mock_executor.run.return_value = Mock(
        returncode=0, stderr="", stdout=""
    )
    result = self.runner.run_export("op1")
    self.assertEqual(result.returncode, 0)
    self.mock_executor.run.assert_called_once()

  def test_run_heir_opt_success(self):
    self.mock_executor.run.return_value = Mock(
        returncode=0, stderr="", stdout=""
    )
    result = self.runner.run_heir_opt("op1")
    self.assertEqual(result.returncode, 0)
    self.mock_executor.run.assert_called_once()

  def test_run_all_pass(self):
    self.mock_executor.run.return_value = Mock(
        returncode=0, stderr="", stdout=""
    )
    results = self.runner.run_all()
    self.assertEqual(results["op1"][0], "PASS")

  def test_run_all_fail_export(self):
    self.mock_executor.run.return_value = Mock(
        returncode=1, stderr="Export failed", stdout=""
    )
    results = self.runner.run_all()
    self.assertEqual(results["op1"][0], "FAIL (Export)")

  def test_format_results_table(self):
    results = {"op1": ("PASS", "")}
    table = self.runner.format_results_table(results)
    self.assertIn("op1", table)
    self.assertIn("PASS", table)

  @patch("sys.stdout", new_callable=io.StringIO)
  def test_run_all_no_spam_stdout(self, mock_stdout):
    self.mock_executor.run.return_value = Mock(
        returncode=0, stderr="", stdout=""
    )
    self.runner.run_all()
    self.assertNotIn("Running coverage for", mock_stdout.getvalue())


if __name__ == "__main__":
  unittest.main()
