import os
from pathlib import Path

from lit.formats import ShTest

config.name = "heir"
config.test_format = ShTest()
config.suffixes = [".mlir", ".v"]

# lit executes relative to the directory
#
#   bazel-bin/tests/<test_target_name>.runfiles/heir/
#
# which contains tools/ and tests/ directories and the binary targets built
# within them, brought in via the `data` attribute in the BUILD file. To
# manually inspect the filesystem in situ, add the following to this script and
# run `bazel test //tests:<target>`
#
#   import subprocess
#
#   print(subprocess.run(["pwd",]).stdout)
#   print(subprocess.run(["ls", "-l", os.environ["RUNFILES_DIR"]]).stdout)
#   print(subprocess.run([ "env", ]).stdout)
#
# Hence, to get lit to see tools like `heir-opt`, we need to add the tools/
# subdirectory to the PATH environment variable.
#
# Bazel defines RUNFILES_DIR which includes heir/ and third party dependencies
# as their own directory. Generally, it seems that $PWD == $RUNFILES_DIR/heir/

runfiles_dir = Path(os.environ["RUNFILES_DIR"])

mlir_tools_relpath = "llvm-project/mlir"
mlir_tools_path = runfiles_dir.joinpath(Path(mlir_tools_relpath))
tool_relpaths = [
    mlir_tools_relpath,
    "heir/tools",
    "heir/tests/verilog",
    "llvm-project/llvm",
    "at_clifford_yosys",
]

CMAKE_HEIR_PATH = os.environ.get("CMAKE_HEIR_PATH","")
if CMAKE_HEIR_PATH:
    CMAKE_HEIR_PATH = ":"+CMAKE_HEIR_PATH
config.environment["PATH"] = (
    ":".join(str(runfiles_dir.joinpath(Path(path))) for path in tool_relpaths)
    + CMAKE_HEIR_PATH
    + ":"
    + os.environ["PATH"]
)

abc_relpath = "edu_berkeley_abc/abc"
config.environment["HEIR_ABC_BINARY"] = (
    str(runfiles_dir.joinpath(Path(abc_relpath)))
)
yosys_libs = "heir/lib/Transforms/YosysOptimizer/yosys"
config.environment["HEIR_YOSYS_SCRIPTS_DIR"] = (
    str(runfiles_dir.joinpath(Path(yosys_libs)))
)

# Some tests that use mlir-cpu-runner need access to additional shared libs to
# link against functions like print. Substitutions replace magic strings in the
# test files with the needed paths.
substitutions = {
    "%mlir_lib_dir": str(mlir_tools_path),
    "%shlibext": ".so",
    "%mlir_runner_utils": os.path.join(
        mlir_tools_path, "libmlir_runner_utils.so"
    ),
}
config.substitutions.extend(substitutions.items())
