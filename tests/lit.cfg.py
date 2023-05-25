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
#   print(subprocess.run(["ls", "-l", "."]).stdout)
#   print(subprocess.run([ "env", ]).stdout)
#
# Hence, to get lit to see tools like `heir-opt`, we need to add the tools/
# subdirectory to the PATH environment variable.
#
# Bazel defines RUNFILES_DIR which includes heir/ and third party dependencies
# as their own directory. Generally, it seems that $PWD == $RUNFILES_DIR/heir/

heir_tools_relpath = Path("heir/tools")
mlir_tools_relpath = Path("llvm-project/mlir")
llvm_tools_relpath = Path("llvm-project/llvm")
runfiles_dir = Path(os.environ["RUNFILES_DIR"])

heir_tools_path = runfiles_dir.joinpath(heir_tools_relpath)
mlir_tools_path = runfiles_dir.joinpath(mlir_tools_relpath)
llvm_tools_path = runfiles_dir.joinpath(llvm_tools_relpath)
tool_paths = [heir_tools_path, mlir_tools_path, llvm_tools_path]

config.environment["PATH"] = (
    ":".join(str(x) for x in tool_paths) + ":" + os.environ["PATH"]
)

# Some tests that use mlir-cpu-runner need access to additional shared libs to
# link against functions like print. Substitutions replace magic strings in the
# test files with the needed paths.
substitutions = {
    "%mlir_lib_dir": mlir_tools_path,
    "%shlibext": ".so",
    "%mlir_runner_utils": os.path.join(
        mlir_tools_path, "libmlir_runner_utils.so"
    ),
}
config.substitutions.extend(substitutions.items())
