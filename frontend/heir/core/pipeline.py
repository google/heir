"""The compilation pipeline."""

import dataclasses
import importlib
import pathlib
import shutil
import sys
import tempfile

from typing import Optional

from heir.backend import clang
from heir.backend import pybind_helpers
from heir.backend.openfhe import openfhe_config as openfhe_config_lib
from heir.core import heir_cli
from frontend.heir.core import heir_cli_config as heir_config_lib
from heir.core import mlir_emitter
from heir.mlir.types import MLIRTypeAnnotation, Secret, Tensor

# FIXME: Don't use implementation detail _GenericAlias!!!
from typing import get_args, get_origin, _GenericAlias

from numba.core import compiler
from numba.core import sigutils
from numba.core.registry import cpu_target
from numba.core.typed_passes import type_inference_stage

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

  # A list of indices of secret args
  secret_args: list[int]

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
    scheme,
    backend,
    openfhe_config: OpenFHEConfig = openfhe_config_lib.DEFAULT_INSTALLED_OPENFHE_CONFIG,
    heir_config: HEIRConfig = heir_config_lib.DEVELOPMENT_HEIR_CONFIG,
    debug: bool = False,
    heir_opt_options : list[str] = None,
):
  """Run the compiler."""
  # The temporary workspace dir is so that heir-opt, heir-translate, and
  # clang can have places to write their output files. It is cleaned up once
  # the function returns, at which point the compiled python module has been
  # loaded into memory and the raw files are not needed.
  workspace_dir = tempfile.mkdtemp()
  try:
    ssa_ir = compiler.run_frontend(function)

    ##### (Numba) Type Inference
    # Fetch the function's type annotation
    annotation = function.__annotations__
    # Convert those annotations back to a numba signature
    signature = ""
    secret_args = []
    # FIXME: this is a mess, there should probably be recursion here instead!
    for idx, (_, arg_type) in enumerate(annotation.items()):
        if (isinstance(arg_type, _GenericAlias)):
            wrapper = get_origin(arg_type)
            if(issubclass(wrapper, Secret)):
                inner_type = get_args(arg_type)[0]
                if(isinstance(inner_type, _GenericAlias)):
                    args = get_args(inner_type)
                    element_type = args[len(args) - 1].numba_str()
                    #FIXME: Add support for static tensor sizes!
                    signature += f"{element_type}[{','.join([':'] * (len(args) - 1))}],"
                else:
                    signature += f"{get_args(arg_type)[0].numba_str()},"
                secret_args.append(idx)
            elif(issubclass(wrapper, Tensor)):
                args = get_args(arg_type)
                inner_type = args[len(args) - 1].numba_str()
                #FIXME: Add support for static tensor sizes!
                signature += f"{inner_type}[{','.join([':'] * (len(args) - 1))}]"
            else:
                raise ValueError(f"Unsupported type annotation {arg_type}")

        elif (not issubclass(arg_type, MLIRTypeAnnotation)):
            raise ValueError(f"Unsupported type annotation {arg_type}")
        else:
            signature += f"{arg_type.numba_str()},"

    # Set up inference contexts
    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    typingctx.refresh()
    targetctx.refresh()
    fn_args, fn_retty = sigutils.normalize_signature(signature)
    # Run actual inference. TODO(#1162): handle type inference errors
    typemap, restype, calltypes, errs = type_inference_stage(typingctx, targetctx, ssa_ir, fn_args,
                                    None)

    mlir_textual = mlir_emitter.TextualMlirEmitter(ssa_ir, secret_args, typemap, restype).emit()
    func_name = ssa_ir.func_id.func_name
    module_name = f"_heir_{func_name}"

    if(debug):
        mlir_in_filepath = Path(workspace_dir) / f"{func_name}.in.mlir"
        print(f"HEIRpy Debug: Writing input MLIR to \t \t {mlir_in_filepath}")
        with open(mlir_in_filepath, "w") as f:
            f.write(mlir_textual)

    heir_opt = heir_cli.HeirOptBackend(heir_config.heir_opt_path)
    if(debug):
        # Print type annotated version of the input
        mlirpath = Path(workspace_dir) / f"{func_name}.annotated.mlir"
        graphpath = Path(workspace_dir) / f"{func_name}.annotated.dot"
        heir_opt_output, graph = heir_opt.run_binary_stderr(
            input=mlir_textual,
            options=["--annotate-secretness", "--view-op-graph"],
        )
        print(f"HEIRpy Debug: Writing secretness-annotated MLIR to \t {mlirpath}")
        with open(mlirpath, "w") as f:
            f.write(heir_opt_output)

        print(f"HEIRpy Debug: Writing annotated graph to \t {graphpath}")
        with open(graphpath, "w") as f:
            f.write(graph)

    if (heir_opt_options is None and backend == "openfhe"):
        heir_opt_options = [
            f"--mlir-to-{scheme}",
            f"--scheme-to-openfhe=entry-function={func_name}"
        ]
    elif (heir_opt_options is None and backend == "heracles"):
        heir_opt_options = [f"--mlir-to-{scheme}"]
    elif (heir_opt_options is None and backend is None):
        heir_opt_options = [f"--mlir-to-{scheme}"]
    if(debug):
        heir_opt_options.append("--view-op-graph")
        print(f"HEIRpy Debug:\033[1m Running heir-opt {' '.join(heir_opt_options)}\033[0m")
    heir_opt_output, graph = heir_opt.run_binary_stderr(
        input=mlir_textual,
        options=heir_opt_options,
    )

    if(debug):
        # Print output after heir_opt:
        mlirpath = Path(workspace_dir) / f"{func_name}.out.mlir"
        graphpath = Path(workspace_dir) / f"{func_name}.out.dot"
        print(f"HEIRpy Debug: Writing output MLIR to \t \t {mlirpath}")
        with open(mlirpath, "w") as f:
            f.write(heir_opt_output)
        print(f"HEIRpy Debug: Writing output graph to \t \t {graphpath}")
        with open(graphpath, "w") as f:
            f.write(graph)

    heir_translate = heir_cli.HeirTranslateBackend(
        binary_path=heir_config.heir_translate_path
    )

    if (backend == "openfhe"):
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
        if(debug):
            args = clang_backend.clang_arg_helper(cpp_source_filepath=cpp_filepath,
            shared_object_output_filepath=so_filepath,
            include_paths=openfhe_config.include_dirs,
            linker_search_paths=linker_search_paths,
            link_libs=openfhe_config.link_libs)
            print(
            f"HEIRpy Debug:\033[1m Running clang {' '.join(str(arg) for arg in args)}\033[0m"
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
    elif(backend=="heracles"):
        csv_filepath = Path(workspace_dir) / f"{func_name}.csv"
        options=[
                "--emit-heracles-bgv",
                "-o",
                csv_filepath,
            ]
        if(debug):
            print(f"HEIRpy Debug:\033[1m Running heir-translate {' '.join(str(opt) for opt in options)}\033[0m")
        heir_translate.run_binary(
            input=heir_opt_output,
            options=options,
        )
        if(debug):
            print(f"HEIRpy Debug: Wrote CSV to {csv_filepath}")
        return # TODO: Once we can actually generate a python module to interact with SDK, this is where this should happen
    elif(debug and backend==None):
        print(f"HEIRPy Debug: Skipping backend related compilation.")
        return
    else:
        raise ValueError(f"Unsupported backend {backend}")
  finally:
    if debug:
      print(
          f"HEIRpy Debug: Leaving workspace_dir {workspace_dir} for"
          " manual inspection.\n"
      )
    else:
      shutil.rmtree(workspace_dir)
  result = CompilationResult(
      module=bound_module,
      func_name=func_name,
      secret_args=secret_args,
      arg_names=ssa_ir.func_id.arg_names,
      arg_enc_funcs={
          arg_name: getattr(bound_module, f"{func_name}__encrypt__arg{i}")
          for i, arg_name in enumerate(ssa_ir.func_id.arg_names)
          if i in secret_args
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
