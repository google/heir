"""Entrypoint script for Torch-MLIR coverage CI.

This script finds all Torch model scripts in the `models` directory,
runs them to export to MLIR, and then runs `heir-opt` on the generated
MLIR files to check for success or crashes. It generates a summary table
and saves failure logs.
"""

import os
import pathlib
import re
import subprocess
import sys


class CommandExecutor:
  """Executes shell commands, separated by an interface for easier mocking."""

  def run(self, cmd, cwd=None):
    return subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


class CoverageRunner:
  """Runs coverage tests for all models."""

  def __init__(
      self,
      executor=None,
      models_dir=None,
      heir_opt_path="bazel-bin/tools/heir-opt",
  ):
    self.executor = executor or CommandExecutor()
    self.models_dir = (
        models_dir or pathlib.Path(__file__).parent.resolve() / "models"
    )
    self.heir_opt_path = heir_opt_path

  def find_models(self):
    return [
        f.stem
        for f in self.models_dir.iterdir()
        if f.is_file() and f.suffix == ".py"
    ]

  def run_export(self, model_name):
    model_file = self.models_dir / f"{model_name}.py"
    repo_base = self.models_dir.parent.parent.parent.resolve()
    cmd = f"PYTHONPATH={repo_base} {sys.executable} {model_file}"
    result = self.executor.run(cmd, cwd=str(self.models_dir))
    return result

  def run_heir_opt(self, model_name):
    input_path = self.models_dir / f"{model_name}.mlir"
    if not input_path.exists():
      return None

    passes = (
        '--secretize "--annotate-module=backend=lattigo scheme=ckks"'
        ' "--torch-linalg-to-ckks=ciphertext-degree=1024" --scheme-to-lattigo'
    )
    cmd = f"{self.heir_opt_path} {passes} {input_path}"
    result = self.executor.run(cmd)
    return result

  def run_all(self):
    models = self.find_models()
    results = {}
    for model in sorted(models):
      print(f"Running coverage for {model}...", file=sys.stderr)
      export_result = self.run_export(model)
      if export_result.returncode != 0:
        results[model] = ("FAIL (Export)", export_result.stderr)
        self.save_failure_log(
            model, "export", export_result.stderr, export_result.stdout
        )
        continue

      heir_opt_result = self.run_heir_opt(model)
      if heir_opt_result is None:
        results[model] = (
            "FAIL (No MLIR)",
            "torch-mlir failed to produce model.mlir",
        )
        self.save_failure_log(
            model, "heir-opt", "torch-mlir failed to produce model.mlir"
        )
        continue

      if heir_opt_result.returncode != 0:
        results[model] = ("FAIL (heir-opt)", heir_opt_result.stderr)
        self.save_failure_log(
            model, "heir-opt", heir_opt_result.stderr, heir_opt_result.stdout
        )
      else:
        results[model] = ("PASS", "")

    return results

  def save_failure_log(self, model_name, stage, content, stdout=None):
    logs_dir = self.models_dir.parent / "failure_logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"{model_name}_{stage}.log"
    with open(log_file, "w") as f:
      if stdout:
        f.write("=== STDOUT ===\n")
        f.write(stdout)
        f.write("\n=== STDERR ===\n")
      f.write(content)

  def format_results_table(self, results):
    table = "<!-- TORCH_COVERAGE_TABLE -->\n\n"
    table += "| Operator | Status | Details |\n"
    table += "| --- | --- | --- |\n"
    for model, (status, details) in results.items():
      if details:
        if "PLEASE submit a bug report to" in details:
          details_str = "(stack trace)"
        else:
          details_str = details.replace("\n", " ")
          details_str = re.sub(
              r"^.*?\.mlir:\d+:\d+: error:", "error:", details_str
          )
          details_str = re.sub(r"^<unknown>:\d+: error:", "error:", details_str)
          details_str = re.sub(r"^.*?\.mlir: error:", "error:", details_str)
          details_str = details_str[:100]
      else:
        details_str = ""
      table += f"| {model} | {status} | {details_str} |\n"
    return table


def main():
  runner = CoverageRunner()
  results = runner.run_all()
  table = runner.format_results_table(results)
  print(table)
  if "GITHUB_STEP_SUMMARY" in os.environ:
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
      f.write("\n## Torch-MLIR Operator Coverage\n")
      f.write(table)


if __name__ == "__main__":
  main()
