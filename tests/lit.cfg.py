import os
from pathlib import Path

from lit.formats import ShTest

config.name = "heir"
config.test_format = ShTest()
config.suffixes = [".mlir", ".v"]

# lit executes relative to the directory
#
#   bazel-bin/tests/<test_target_name>.runfiles/_main/
#
# which contains tools/ and tests/ directories and the binary targets built
# within them, brought in via the `data` attribute in the BUILD file. To
# manually inspect the filesystem in situ, add the following to this script and
# run `bazel test //tests:<target>`
#
# import subprocess
#
# print(subprocess.run(["pwd",]).stdout)
# print(subprocess.run(["ls", "-l", os.environ["RUNFILES_DIR"]]).stdout)
# print(subprocess.run([ "env", ]).stdout)
#
# Hence, to get lit to see tools like `heir-opt`, we need to add the tools/
# subdirectory to the PATH environment variable.
#
# Bazel defines RUNFILES_DIR which includes _main/ and third party dependencies
# as their own directory. Generally, it seems that $PWD == $RUNFILES_DIR/_main/

runfiles_dir = Path(os.environ["RUNFILES_DIR"])

llvm_project_canonical_name = "+_repo_rules+llvm-project"
mlir_tools_relpath = llvm_project_canonical_name + "/mlir"
llvm_tools_relpath = llvm_project_canonical_name + "/llvm"
mlir_tools_path = runfiles_dir.joinpath(Path(mlir_tools_relpath))
tool_relpaths = [
    mlir_tools_relpath,
    llvm_tools_relpath,
    "_main/tools",
    "_main/tests/Emitter/verilog",
    "yosys+",
]

config.environment["PATH"] = (
    ":".join(str(runfiles_dir.joinpath(Path(path))) for path in tool_relpaths)
    + ":"
    + os.environ["PATH"]
)

abc_relpath = "abc+/abc_bin"
config.environment["HEIR_ABC_BINARY"] = str(
    runfiles_dir.joinpath(Path(abc_relpath))
)
yosys_libs = "_main/lib/Transforms/YosysOptimizer/yosys"
config.environment["HEIR_YOSYS_SCRIPTS_DIR"] = str(
    runfiles_dir.joinpath(Path(yosys_libs))
)

# Some tests that use mlir-runner need access to additional shared libs to
# link against functions like print. Substitutions replace magic strings in the
# test files with the needed paths.
substitutions = {
    "%mlir_lib_dir": str(mlir_tools_path),
}
config.substitutions.extend(substitutions.items())
