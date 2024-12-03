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
    "/usr/local/include/openfhe/core",
    "/usr/local/include/openfhe/pke",
]
OPENFHE_LINK_LIBS = [
    "OPENFHEpke",
    "OPENFHEcore",
]


def run_compiler(function):
    with tempfile.TemporaryDirectory() as tmpdir:
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
        # TODO: construct heir-translate pipeline options from decorator
        openfhe_pke_header = heir_translate.run_binary(
            input=heir_opt_output,
            options=["--emit-openfhe-pke-header"],
        )
        openfhe_pke_source = heir_translate.run_binary(
            input=heir_opt_output,
            options=["--emit-openfhe-pke"],
        )

        clang = ClangBackend()
        clang.compile_to_shared_object(
            openfhe_pke_source,
            shared_object_output_filepath=Path(tmpdir) / f"{func_id.func_name}.so",
            include_paths=OPENFHE_INCLUDE_PATHS,
            link_libs=OPENFHE_LINK_LIBS,
            temp_cpp_filepath=Path(tmpdir) / f"{func_id.func_name}.cpp",
        )

    return openfhe_pke_source, openfhe_pke_header
