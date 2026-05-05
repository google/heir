"""Library for converting e2e tests to bazel run commands."""

import os
import shlex
import subprocess
import xml.etree.ElementTree as ET

DEFAULT_TOOL_PREFIX = "bazel run --noallow_analysis_cache_discard //tools"


def run_blaze_query(query_str, options=None):
  """Runs a blaze query and returns the output."""
  cmd = ["bazel", "query"]
  if options:
    cmd.extend(options)
  cmd.extend([query_str, "--keep_going"])
  cwd = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  try:
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False, cwd=cwd
    )
    # Bazel query returns 3 if it encountered errors but still produced results with --keep_going.
    if result.returncode not in [0, 3]:
      print(f"Warning: blaze query failed with exit code {result.returncode}")
      print(result.stderr)
    return result.stdout
  except Exception as e:
    print(f"Error running blaze query: {e}")
    return ""


def path_to_label(path):
  """Attempts to convert a file path to a bazel label."""
  if path.startswith("//"):
    return path
  if os.path.isfile(path):
    dir_path = os.path.dirname(path)
    file_name = os.path.basename(path)
    if os.path.exists(os.path.join(dir_path, "BUILD")):
      return f"//{dir_path}:{file_name}"
  return path


def get_heir_opt_target(target_or_path):
  """Finds the heir_opt target associated with the given target or path."""
  # Normalize path to label if it looks like a file
  label = path_to_label(target_or_path)

  if label.endswith(".mlir"):
    query = f"kind(heir_opt, rdeps(//tests/Examples/..., {label}))"  # fmt: skip
    output = run_blaze_query(query)
    lines = output.strip().split("\n")
    targets = [l for l in lines if l.startswith("//")]
    if targets:
      return targets[0]
    return None

  if os.path.isdir(target_or_path):
    label = target_or_path
    if not label.startswith("//"):
      label = "//" + label
    query = f"kind(heir_opt, {label}:*)"
    output = run_blaze_query(query)
    lines = output.strip().split("\n")
    targets = [l for l in lines if l.startswith("//")]
    if targets:
      return targets[0]
    return None

  if target_or_path.startswith("//"):
    # Try to find heir_opt targets in the same package first
    pkg = target_or_path.split(":")[0]
    query = f"kind(heir_opt, {pkg}:*)"
    output = run_blaze_query(query)
    lines = output.strip().split("\n")
    targets = [l for l in lines if l.startswith("//")]
    if targets:
      return targets[0]

    # Fallback: search deps
    query = f"kind(heir_opt, deps({target_or_path}))"
    output = run_blaze_query(query)
    lines = output.strip().split("\n")
    targets = [l for l in lines if l.startswith("//")]
    if targets:
      return targets[0]
    return None

  return None


def e2e_to_bazel(target_or_path, tool_prefix=DEFAULT_TOOL_PREFIX):
  """Converts an e2e test target or path to a blaze run command for heir-opt.

  Args:
    target_or_path: The test target, directory, or source file.
    tool_prefix: The prefix for the heir-opt tool.
  """
  heir_opt_target = get_heir_opt_target(target_or_path)
  if not heir_opt_target:
    print(f"Could not find heir_opt target for {target_or_path}")
    return

  xml_output = run_blaze_query(heir_opt_target, options=["--output=xml"])

  if not xml_output:
    print("Failed to get XML output from blaze query")
    return

  try:
    root = ET.fromstring(xml_output)
    rule = root.find("rule")
    if rule is None or rule.get("class") != "heir_opt":
      print(f"Target {heir_opt_target} is not an heir_opt rule")
      return

    pass_flags = []
    src = ""

    for list_elem in rule.findall("list"):
      if list_elem.get("name") == "pass_flags":
        for str_elem in list_elem.findall("string"):
          pass_flags.append(str_elem.get("value"))

    for label_elem in rule.findall("label"):
      if label_elem.get("name") == "src":
        src = label_elem.get("value")

    if not src:
      print("Could not find src attribute")
      return

    # Resolve src label to path
    src_path = src
    if src_path.startswith("//"):
      src_path = src_path[2:].replace(":", "/")

      workspace_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
      if not workspace_root:
        try:
          result = subprocess.run(
              ["bazel", "info", "workspace"],
              capture_output=True,
              text=True,
              check=True,
          )
          workspace_root = result.stdout.strip()
        except Exception as e:
          # Fallback to relative path if we can't get workspace root
          workspace_root = ""

      if workspace_root:
        src_path = os.path.join(workspace_root, src_path)

    flags_str = " ".join(shlex.quote(f) for f in pass_flags)
    command = f"{tool_prefix}:heir-opt -- {flags_str} {src_path}"
    print(command)

  except Exception as e:
    print(f"Error parsing XML: {e}")
