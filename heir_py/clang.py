"""A helper to fing and run clang."""

import os
import subprocess
import tempfile
from pathlib import Path


DEFAULT_COMPILER_FLAGS = "-O3 -fPIC -shared -stdlib=libc++ -std=c++17".split()


def to_clang_args(prefix, strs):
    return [f"{prefix}{s}" for s in strs] if strs else []


class ClangBackend:
    """A helper to find and run clang."""

    def __init__(self, path_to_clang: str = None):
        self.compiler_binary_path = path_to_clang or self._find_clang()

    def _find_clang(self):
        # FIXME: can I use $CXX, or else just allow GCC to be used in place of clang?
        clang = os.environ.get("CLANG")
        if clang is not None:
            return clang

        try:
            clang = subprocess.check_output(["which", "clang"]).strip().decode()
            return clang
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "Could not find clang executable. It should be passed to"
                "ClangBackend via path_to_clang, or specified in the CLANG"
                "environment variable, or be on the PATH."
            )

    def compile_to_shared_object(
        self,
        cpp_source: str,
        shared_object_output_filepath: Path,
        compiler_flags: str = DEFAULT_COMPILER_FLAGS,
        include_paths: list[str] = None,
        linker_search_paths: list[str] = None,
        link_libs: list[str] = None,
        temp_cpp_filepath: str = None,
    ) -> Path:
        """Compile C++ source to a shared object file.

        Args:
            cpp_source: the C++ source code to compile
            shared_object_output_filepath: the path to the output .so file
            compiler_flags:
            include_paths: list[str] = None,
            linker_search_paths: list[str] = None,
            link_libs: list[str] = None,
            temp_cpp_filepath: str = None,
        """
        # err if output filepath does not end with .so
        if shared_object_output_filepath.suffix != ".so":
            raise ValueError(
                f"Expected shared object output filepath to end with .so, but got {shared_object_output_filepath}"
            )

        with open(temp_cpp_filepath, mode="w") as f:
            f.write(cpp_source)

        include_args = to_clang_args("-I", include_paths)
        linker_search_path_args = to_clang_args("-L", linker_search_paths)
        link_lib_args = to_clang_args("-l", link_libs)
        args = (
            [
                self.compiler_binary_path,
                temp_cpp_filepath,
                "-o",
                shared_object_output_filepath,
            ]
            + compiler_flags
            + include_args
            + linker_search_path_args
            + link_lib_args
        )
        completed_process = subprocess.run(
            args,
            text=True,
            capture_output=True,
        )
        if completed_process.returncode != 0:
            message = f"Error running {self.compiler_binary_path}. "
            message += "stderr was:\n\n" + completed_process.stderr + "\n\n"
            message += "options were:\n\n" + " ".join([str(x) for x in args])
            raise ValueError(message)
