"""The compilation pipeline."""

import dataclasses
import importlib
import pathlib
import shutil
import sys
import tempfile

from heir_py import clang
from heir_py import heir_backend
from heir_py import heir_config as heir_config_lib
from heir_py import mlir_emitter
from heir_py import openfhe_config as openfhe_config_lib
from heir_py import pybind_helpers
from numba.core import bytecode
from numba.core import interpreter

dataclass = dataclasses.dataclass
Path = pathlib.Path
pyconfig_ext_suffix = pybind_helpers.pyconfig_ext_suffix
OpenFHEConfig = openfhe_config_lib.OpenFHEConfig
pybind11_includes = pybind_helpers.pybind11_includes
HEIRConfig = heir_config_lib.HEIRConfig


@dataclass
class CompilationResult:
  # The module object containing the compiled functions
  module: object

  # The function name used to generate the various compiled functions
  func_name: str

  # A list of arg names (in order)
  arg_names: list[str]

  # A mapping from argument name to the compiled encryption function
  arg_enc_funcs: dict[str, object]

  # The compiled decryption function for the function result
  result_dec_func: object

  # The main compiled function
  main_func: object

  # Backend setup functions, if any
  setup_funcs: dict[str, object]


def run_compiler(
    function,
    openfhe_config: OpenFHEConfig = openfhe_config_lib.DEFAULT_INSTALLED_OPENFHE_CONFIG,
    heir_config: HEIRConfig = heir_config_lib.DEVELOPMENT_HEIR_CONFIG,
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
    func_id = bytecode.FunctionIdentity.from_function(function)
    converted_bytecode = bytecode.ByteCode(func_id)
    ssa_ir = interpreter.Interpreter(func_id).interpret(converted_bytecode)
    mlir_textual = mlir_emitter.TextualMlirEmitter(ssa_ir).emit()
    func_name = func_id.func_name
    module_name = f"_heir_{func_name}"

    heir_opt = heir_backend.HeirOptBackend(heir_config.heir_opt_path)
    # TODO(#1162): construct heir-opt pipeline options from decorator
    heir_opt_options = [
        # Canonicalize cleans up the emitted IR
        "--canonicalize",
        f"--secretize=function={func_name}",
        "--mlir-to-bgv=ciphertext-degree=32",
        f"--scheme-to-openfhe=entry-function={func_name}",
    ]
    heir_opt_output = heir_opt.run_binary(
        input=mlir_textual,
        options=heir_opt_options,
    )

    if debug:
      mlir_in_filepath = Path(workspace_dir) / f"{func_name}.in.mlir"
      print(f"Debug mode enabled. Writing Input MLIR to {mlir_in_filepath}")
      with open(mlir_in_filepath, "w") as f:
        f.write(mlir_textual)
      mlir_out_filepath = Path(workspace_dir) / f"{func_name}.out.mlir"
      print(f"Debug mode enabled. Writing Output MLIR to {mlir_out_filepath}")
      with open(mlir_out_filepath, "w") as f:
        f.write(heir_opt_output)

    heir_translate = heir_backend.HeirTranslateBackend(
        binary_path=heir_config.heir_translate_path
    )
    cpp_filepath = Path(workspace_dir) / f"{func_name}.cpp"
    h_filepath = Path(workspace_dir) / f"{func_name}.h"
    pybind_filepath = Path(workspace_dir) / f"{func_name}_bindings.cpp"
    # TODO(#1162): construct heir-translate pipeline options from decorator
    include_type_flag = "--openfhe-include-type=" + openfhe_config.include_type
    heir_translate.run_binary(
        input=heir_opt_output,
        options=[
            "--emit-openfhe-pke-header",
            include_type_flag,
            "-o",
            h_filepath,
        ],
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

    clang_backend = clang.ClangBackend()
    so_filepath = Path(workspace_dir) / f"{func_name}.so"
    linker_search_paths = [openfhe_config.lib_dir]
    if debug:
      print(
          f"Debug mode enabled. Compiling {cpp_filepath} with linker search"
          f" paths: {linker_search_paths},  include paths:"
          f" {openfhe_config.include_dirs}, link libs:"
          f" {openfhe_config.link_libs}\n"
      )

    clang_backend.compile_to_shared_object(
        cpp_source_filepath=cpp_filepath,
        shared_object_output_filepath=so_filepath,
        include_paths=openfhe_config.include_dirs,
        linker_search_paths=linker_search_paths,
        link_libs=openfhe_config.link_libs,
    )

    ext_suffix = pyconfig_ext_suffix()
    pybind_so_filepath = Path(workspace_dir) / f"{module_name}{ext_suffix}"
    clang_backend.compile_to_shared_object(
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
  except Exception as e:
    print(f"error: {e}")
  finally:
    if debug:
      print(
          f"Debug mode enabled. Leaving workspace_dir {workspace_dir} for"
          " manual inspection."
      )
    else:
      shutil.rmtree(workspace_dir)

  result = CompilationResult(
      module=bound_module,
      func_name=func_name,
      arg_names=func_id.arg_names,
      arg_enc_funcs={
          arg_name: getattr(bound_module, f"{func_name}__encrypt__arg{i}")
          for i, arg_name in enumerate(func_id.arg_names)
      },
      result_dec_func=getattr(bound_module, f"{func_name}__decrypt__result0"),
      main_func=getattr(bound_module, func_name),
      setup_funcs={
          "generate_crypto_context": getattr(
              bound_module, f"{func_name}__generate_crypto_context"
          ),
          "configure_crypto_context": getattr(
              bound_module, f"{func_name}__configure_crypto_context"
          ),
      },
  )

  return result
