"""The compilation pipeline."""

from pathlib import Path
import importlib
import shutil
import sys
import tempfile

from heir_py.openfhe_config import OpenFHEConfig, DEFAULT_INSTALLED_OPENFHE_CONFIG
from heir_py.heir_config import HEIRConfig, DEVELOPMENT_HEIR_CONFIG
from heir_py.mlir_emitter import TextualMlirEmitter
from heir_py.pybind_helpers import pybind11_includes, pyconfig_ext_suffix
from numba.core.bytecode import ByteCode, FunctionIdentity
from numba.core.interpreter import Interpreter
from heir_py.heir_backend import HeirOptBackend, HeirTranslateBackend
from heir_py.clang import ClangBackend


def run_compiler(
    function,
    openfhe_config: OpenFHEConfig = DEFAULT_INSTALLED_OPENFHE_CONFIG,
    heir_config: HEIRConfig = DEVELOPMENT_HEIR_CONFIG,
    debug=False,
):
    """Run the compiler."""
    # The temporary workspace dir is so that heir-opt, heir-translate, and
    # clang can have places to write their output files. It is cleaned up once
    # the function returns, at which point the compiled python module has been
    # loaded into memory and the raw files are not needed.
    #
    # For debugging, add delete=False to TemporaryDirectory (python3.12+)
    # to leave the directory around after the context manager closes.
    # Otherwise, replace the context manager with `workspace_dir =
    # tempfile.mkdtemp()` and manually clean it up.
    workspace_dir = tempfile.mkdtemp()
    try:
        func_id = FunctionIdentity.from_function(function)
        bytecode = ByteCode(func_id)
        ssa_ir = Interpreter(func_id).interpret(bytecode)
        mlir_textual = TextualMlirEmitter(ssa_ir).emit()
        func_name = func_id.func_name
        module_name = f"_heir_{func_name}"

        heir_opt = HeirOptBackend(binary_path=heir_config.heir_opt_path)
        # TODO(#1162): construct heir-opt pipeline options from decorator
        heir_opt_options = [
            f"--secretize=function={func_name}",
            "--mlir-to-openfhe-bgv="
            f"entry-function={func_name} ciphertext-degree=32",
        ]
        heir_opt_output = heir_opt.run_binary(
            input=mlir_textual,
            options=heir_opt_options,
        )

        heir_translate = HeirTranslateBackend(binary_path=heir_config.heir_translate_path)
        cpp_filepath = Path(workspace_dir) / f"{func_name}.cpp"
        h_filepath = Path(workspace_dir) / f"{func_name}.h"
        pybind_filepath = Path(workspace_dir) / f"{func_name}_bindings.cpp"
        # TODO(#1162): construct heir-translate pipeline options from decorator
        include_type_flag = "--openfhe-include-type=" + openfhe_config.include_type
        heir_translate.run_binary(
            input=heir_opt_output,
            options=["--emit-openfhe-pke-header", include_type_flag, "-o", h_filepath],
        )
        heir_translate.run_binary(
            input=heir_opt_output,
            options=["--emit-openfhe-pke", include_type_flag, "-o", cpp_filepath],
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
        so_filepath = Path(workspace_dir) / f"{func_name}.so"
        linker_search_paths = [openfhe_config.lib_dir]
        clang.compile_to_shared_object(
            cpp_source_filepath=cpp_filepath,
            shared_object_output_filepath=so_filepath,
            include_paths=openfhe_config.include_dirs,
            linker_search_paths=linker_search_paths,
            link_libs=openfhe_config.link_libs,
        )

        ext_suffix = pyconfig_ext_suffix()
        pybind_so_filepath = Path(workspace_dir) / f"{module_name}{ext_suffix}"
        clang.compile_to_shared_object(
            cpp_source_filepath=pybind_filepath,
            shared_object_output_filepath=pybind_so_filepath,
            include_paths=openfhe_config.include_dirs
            + pybind11_includes()
            + [workspace_dir],
            linker_search_paths=linker_search_paths,
            link_libs=openfhe_config.link_libs,
            linker_args=["-rpath", ":".join(linker_search_paths)],
            abs_link_lib_paths=[so_filepath],
        )

        sys.path.append(workspace_dir)
        importlib.invalidate_caches()
        bound_module = importlib.import_module(module_name)
    finally:
        if debug:
            print(f"Debug mode enabled. Leaving workspace_dir {workspace_dir} for manual inspection.")
        else:
            shutil.rmtree(workspace_dir)

    return bound_module
