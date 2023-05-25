import os

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
#   print(subprocess.run(["pwd", ]).stdout)
#   print(subprocess.run(["ls", "-l", "."]).stdout)
#   print(subprocess.run(["env",]).stdout)
#
# Hence, to get lit to see tools like `heir-opt`, we need to add the tools/
# subdirectory to the PATH environment variable.

heir_tools_path = os.path.join(os.environ["PWD"], "tools")
mlir_tools_path = os.path.join(
    os.environ["RUNFILES_DIR"], "llvm-project", "mlir"
)
llvm_tools_path = os.path.join(
    os.environ["RUNFILES_DIR"], "llvm-project", "llvm"
)
config.environment["PATH"] = (
    ":".join([heir_tools_path, mlir_tools_path, llvm_tools_path])
    + ":"
    + os.environ["PATH"]
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
