"""Benchmark build time for generated C++ code"""

import difflib
import re
import subprocess
import time

levels = [
    ('-O0', '["-O0"]'),
    ('-O1', '["-O1"]'),
    ('-O2', '["-O2"]'),
    ('-O3', '["-O3"]'),
]

results = []

build_target = (
    "//third_party/heir/tests/Examples/openfhe/ckks/mnist:_heir_mnist_openfhe"
)
test_target = "//third_party/heir/tests/Examples/openfhe/ckks/mnist:mnist_test"
build_all = "//third_party/heir/tests/Examples/openfhe/ckks/mnist:all"
build_file = "tests/Examples/openfhe/ckks/mnist/BUILD"


def run_command(cmd):
  start = time.time()
  process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
  end = time.time()
  return end - start, process.stdout + process.stderr


# Ensure prerequisites are built and cleaned
subprocess.run(
  "blaze clean --expunge",
  shell=True,
  capture_output=True,
)
subprocess.run(
  "blaze build -c opt //third_party/heir/tools:all"
  " //third_party/heir/tests:test_utilities",
  shell=True,
  capture_output=True,
)

for label, copt_value in levels:
  print(f"Benchmarking {label}...")

  with open(build_file, "r") as f:
    content = f.read()
  new_content = re.sub(r"copts = \[.*?\],", f"copts = {copt_value},", content)
  diff = difflib.ndiff(content.splitlines(), new_content.splitlines())
  print("Updated build file:")
  print("\n".join(diff))

  with open(build_file, "w") as f:
    f.write(new_content)

  print(f"  Compiling...")
  comp_time, comp_output = run_command(f"blaze build -c opt {build_target}")
  subprocess.run(
      f"blaze build -c opt {build_all}", shell=True, capture_output=True
  )

  print(f"  Testing...")
  test_start = time.time()
  test_proc = subprocess.run(
      f"blaze test -c opt {test_target}",
      shell=True,
      capture_output=True,
      text=True,
  )
  test_end = time.time()

  # The actual test execution time (excluding overhead) is sometimes in the output
  # e.g. //third_party/heir/tests/Examples/openfhe/ckks/mnist:mnist_test  PASSED in 384.8s
  match = re.search(r"mnist_test\s+PASSED in ([\d\.]+)s", test_proc.stdout)
  if match:
    test_exec_time = float(match.group(1))
  else:
    test_exec_time = test_end - test_start

  results.append({
      "Optimization Level": label,
      "Compilation Time (s)": f"{comp_time:.2f}",
      "Test Execution Time (s)": f"{test_exec_time:.2f}",
  })
  print(f"  Result: Comp={comp_time:.2f}s, Test={test_exec_time:.2f}s")

# Generate Markdown table
table = (
    "| Optimization Level | Compilation Time (s) | Test Execution Time (s) |\n"
)
table += "| --- | --- | --- |\n"
for r in results:
  table += (
      f"| {r['Optimization Level']} | {r['Compilation Time (s)']} |"
      f" {r['Test Execution Time (s)']} |\n"
  )

print("\n" + table)

with open("tests/Examples/openfhe/ckks/mnist/benchmark_results.md", "w") as f:
  f.write(table)
