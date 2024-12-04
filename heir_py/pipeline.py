"""The compilation pipeline."""

from pathlib import Path
import importlib
import sys
import tempfile

from heir_py.mlir_emitter import TextualMlirEmitter
from heir_py.pybind_helpers import pybind11_includes, pyconfig_ext_suffix
from numba.core.bytecode import ByteCode, FunctionIdentity
from numba.core.interpreter import Interpreter
from heir_py.heir_backend import HeirOptBackend, HeirTranslateBackend
from heir_py.clang import ClangBackend


# FIXME: better organization of OpenFHE libs and includes
OPENFHE_INCLUDE_PATHS = [
    "/usr/local/include/openfhe",
    "/usr/local/include/openfhe/binfhe",
    "/usr/local/include/openfhe/core",
    "/usr/local/include/openfhe/pke",
]
# The directory containing libOPENFHEbinfhe.so, etc.
OPENFHE_LIB_DIRS = [
    "/usr/local/lib",
]
# The names of the libraries to link against (without lib prefix or .so suffix)
OPENFHE_LINK_LIBS = [
    "OPENFHEbinfhe",
    "OPENFHEcore",
    "OPENFHEpke",
]
# FIXME: figure out how to discover these, and make them overridable by the user
# maybe check $CPLUS_INCLUDE_PATH
LIBCXX_INCLUDES = [
    "/usr/include/c++/11/",
    "/usr/include/x86_64-linux-gnu/c++/11/",
]
LIBCXX_LIBS = [
    "/usr/lib/gcc/x86_64-linux-gnu/11/",
]



def run_compiler(function):
    # FIXME: can we do this entirely with a tmpdir that is deleted after the
    # module has been loaded?
    workspace_dir = tempfile.mkdtemp()

    func_id = FunctionIdentity.from_function(function)
    bytecode = ByteCode(func_id)
    ssa_ir = Interpreter(func_id).interpret(bytecode)
    mlir_textual = TextualMlirEmitter(ssa_ir).emit()
    module_name = f"_heir_{func_id.func_name}"

    # FIXME: allow user to configure heir-opt path
    heir_opt = HeirOptBackend(binary_path="tools/heir-opt")
    # FIXME: construct heir-opt pipeline options from decorator
    heir_opt_options = [
        "--mlir-to-openfhe-bgv="
        f"entry-function={func_id.func_name} ciphertext-degree=32",
    ]
    heir_opt_output = heir_opt.run_binary(
        input=mlir_textual,
        options=heir_opt_options,
    )

    heir_translate = HeirTranslateBackend(binary_path="tools/heir-translate")
    cpp_filepath = Path(workspace_dir) / f"{func_id.func_name}.cpp"
    h_filepath = Path(workspace_dir) / f"{func_id.func_name}.h"
    pybind_filepath = Path(workspace_dir) / f"{func_id.func_name}_bindings.cpp"
    # FIXME: construct heir-translate pipeline options from decorator
    heir_translate.run_binary(
        input=heir_opt_output,
        options=["--emit-openfhe-pke-header", "-o", h_filepath],
    )
    heir_translate.run_binary(
        input=heir_opt_output,
        options=["--emit-openfhe-pke", "-o", cpp_filepath],
    )
    heir_translate.run_binary(
        input=heir_opt_output,
        options=[
            "--emit-openfhe-pke-pybind",
            f"--pybind-header-include={h_filepath.name}",
            f"--pybind-module-name={module_name}",
            "-o",
            pybind_filepath,
        ],
    )

    clang = ClangBackend()
    so_filepath = Path(workspace_dir) / f"lib{func_id.func_name}.so"
    clang.compile_to_shared_object(
        cpp_source_filepath=cpp_filepath,
        shared_object_output_filepath=so_filepath,
        include_paths=OPENFHE_INCLUDE_PATHS + LIBCXX_INCLUDES,
        link_libs=OPENFHE_LINK_LIBS,
    )

    pybind_includes = pybind11_includes()
    ext_suffix = pyconfig_ext_suffix()
    pybind_so_filepath = Path(workspace_dir) / f"{module_name}{ext_suffix}"
    linker_search_paths = [workspace_dir] + OPENFHE_LIB_DIRS + LIBCXX_LIBS
    clang.compile_to_shared_object(
        cpp_source_filepath=pybind_filepath,
        shared_object_output_filepath=pybind_so_filepath,
        include_paths=OPENFHE_INCLUDE_PATHS
        + LIBCXX_INCLUDES
        + pybind_includes
        + [workspace_dir],
        linker_search_paths=linker_search_paths,
        # allow, e.g., libfoo.so to be found via `-lfoo`
        link_libs=OPENFHE_LINK_LIBS + [func_id.func_name],
        linker_args=["-rpath", ":".join(linker_search_paths)],
    )

    sys.path.append(workspace_dir)
    importlib.invalidate_caches()
    bound_module = importlib.import_module(module_name)

    return bound_module
