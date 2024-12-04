"""The compilation pipeline."""
from pathlib import Path
import tempfile

from heir_py.mlir_emitter import TextualMlirEmitter
from numba.core.bytecode import ByteCode, FunctionIdentity
from numba.core.interpreter import Interpreter
from heir_py.heir_backend import HeirOptBackend, HeirTranslateBackend
from heir_py.clang import ClangBackend


# TODO: better organization of OpenFHE libs and includes
OPENFHE_INCLUDE_PATHS = [
    "/usr/local/include/openfhe",
    "/usr/local/include/openfhe/binfhe",
    "/usr/local/include/openfhe/core",
    "/usr/local/include/openfhe/pke",
]
OPENFHE_LINK_LIBS = [
    "OPENFHEbinfhe",
    "OPENFHEcore",
    "OPENFHEpke",
]
# TODO: figure out how to discover these, and make them overridable by the user
LIBCXX_LIBS = [
    "/usr/include/c++/11/",
    "/usr/include/x86_64-linux-gnu/c++/11/",
]


def run_compiler(function):
    with tempfile.TemporaryDirectory() as workspace_dir:
        func_id = FunctionIdentity.from_function(function)
        bytecode = ByteCode(func_id)
        ssa_ir = Interpreter(func_id).interpret(bytecode)
        mlir_textual = TextualMlirEmitter(ssa_ir).emit()

        # FIXME: allow user to configure heir-opt path
        heir_opt = HeirOptBackend(binary_path="tools/heir-opt")
        # TODO: construct heir-opt pipeline options from decorator
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
        # TODO: construct heir-translate pipeline options from decorator
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
                f"--emit-openfhe-pke-pybind=header={h_filepath}",
                "-o",
                pybind_filepath,
            ],
        )

        clang = ClangBackend()
        so_filepath = Path(workspace_dir) / f"{func_id.func_name}.so"
        clang.compile_to_shared_object(
            cpp_source_filepath=cpp_filepath,
            shared_object_output_filepath=so_filepath,
            include_paths=OPENFHE_INCLUDE_PATHS + LIBCXX_LIBS,
            link_libs=OPENFHE_LINK_LIBS,
        )

    return None
