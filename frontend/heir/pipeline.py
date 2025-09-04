"""The compilation pipeline."""

import os
import pathlib
import re
import shutil
import tempfile
from typing import Any, Callable, Optional

from numba.core import compiler
from numba.core import sigutils
from numba.core.ir import FunctionIR
from numba.core.registry import cpu_target
from numba.core.typed_passes import type_inference_stage
from numba.core.types import Type as NumbaType
from numba.core.types.misc import NoneType

from heir.backends.cleartext import CleartextBackend
from heir.backends.openfhe import OpenFHEBackend, config as openfhe_config
from heir.backends.util.common import is_pip_installed
from heir.heir_cli import heir_cli, heir_cli_config
from heir.heir_cli.heir_cli import CLIError
from heir.interfaces import BackendInterface, ClientInterface, CompilerError, DebugMessage, InternalCompilerError
from heir.mlir.types import parse_annotations
from heir.mlir_emitter import TextualMlirEmitter


# TODO (#1312): Allow for multiple functions in the same compilation.


Path = pathlib.Path
HEIRConfig = heir_cli_config.HEIRConfig


_UNIQUE_NAME_COUNTER = 0


def _get_unique_func_name(func: Callable) -> str:
  """Create a unique sanitized function name.

  Create a unique function name from the function's module and qualified name,
  and sanitize it to be a valid C++ identifier. This is used to avoid name
  collisions when compiling multiple functions with the same name.
  """
  global _UNIQUE_NAME_COUNTER
  sanitized = re.sub(
      r"[^a-zA-Z0-9_]", "_", f"{func.__module__}_{func.__qualname__}"
  )
  unique_name = f"{sanitized}_{_UNIQUE_NAME_COUNTER}"
  _UNIQUE_NAME_COUNTER += 1
  return unique_name


def run_pipeline(
    function_or_mlir_str: Callable | str,
    heir_opt_options: list[str],
    backend: BackendInterface,
    heir_config: Optional[HEIRConfig],
    debug: bool,
) -> ClientInterface:
  """Run the pipeline."""
  if not heir_config:
    heir_config = heir_cli_config.from_os_env()

  # Set environment variables from HEIR config
  os.environ["HEIR_ABC_BINARY"] = os.path.abspath(str(heir_config.abc_path))
  os.environ["HEIR_YOSYS_SCRIPTS_DIR"] = os.path.abspath(
      str(heir_config.techmap_dir_path)
  )

  # The temporary workspace dir is so that heir-opt and the backend
  # can have places to write their output files. It is cleaned up once
  # the function returns, at which point the compiled python module has been
  # loaded into memory and the raw files are not needed.
  workspace_dir = tempfile.mkdtemp()
  try:

    # If it's a python function, parse it via Numba/etc
    mlir_raw_textual = ""
    if isinstance(function_or_mlir_str, Callable):
      numba_ir = compiler.run_frontend(function_or_mlir_str)
      if not isinstance(numba_ir, FunctionIR):
        raise InternalCompilerError(
            f"Expected FunctionIR from numba frontend but got {type(numba_ir)}"
        )
      func_name = _get_unique_func_name(function_or_mlir_str)
      arg_names = numba_ir.func_id.arg_names

      # (Numba) Type Inference
      fn_args: list[NumbaType] = []
      secret_args: list[int] = []
      try:
        fn_args, secret_args, rettype = parse_annotations(
            function_or_mlir_str.__annotations__
        )
      except Exception as e:
        raise CompilerError(
            f"Signature parsing failed for function {func_name} with"
            f" {type(e).__name__}: {e}",
            numba_ir.loc,
        )
      typingctx = cpu_target.typing_context
      targetctx = cpu_target.target_context
      typingctx.refresh()
      targetctx.refresh()
      try:
        typemap, restype, _, _ = type_inference_stage(
            typingctx, targetctx, numba_ir, fn_args, None
        )
      except Exception as e:
        raise CompilerError(
            f"Type inference failed for function {func_name} "
            f"with {type(e).__name__}: {e}",
            numba_ir.loc,
        )

      # Check if we found a return type:
      if restype is None or isinstance(restype, NoneType):
        raise CompilerError(
            f"Type inference failed for function {func_name}: "
            "no return type could be determined.",
            numba_ir.loc,
        )

      # If a result type was annotated, compare with numba
      if rettype is not None and debug:
        if rettype != restype:
          DebugMessage(
              " Warning: user provided return type does not match"
              f" numba inference, expected {restype}, got {rettype}",
              numba_ir.loc,
          )

      # Emit Textual IR
      mlir_raw_textual = TextualMlirEmitter(
          numba_ir, secret_args, typemap, restype, func_name
      ).emit()

      if debug:
        mlir_raw_filepath = Path(workspace_dir) / f"{func_name}.raw.mlir"
        DebugMessage(f"Writing raw input MLIR to {mlir_raw_filepath}")
        with open(mlir_raw_filepath, "w") as f:
          f.write(mlir_raw_textual)

    # If the input is a string, use it directly
    elif isinstance(function_or_mlir_str, str):
      mlir_raw_textual = function_or_mlir_str

    # Try to find heir_opt and heir_translate
    heir_opt = heir_cli.HeirOptBackend(heir_config.heir_opt_path)
    heir_translate = heir_cli.HeirTranslateBackend(
        binary_path=heir_config.heir_translate_path
    )

    # If we have a raw MLIR string, we need to ask heir for the function name,
    # argument names, and secret argument indices.
    if isinstance(function_or_mlir_str, str):
      try:
        # We first need to --mlir-print-op-generic so that we can later use
        # --allow-unregistered-dialect with heir_translate
        generic_mlir = heir_opt.run_binary(
            input=mlir_raw_textual, options=["--mlir-print-op-generic"]
        )
        func_info_str = heir_translate.run_binary(
            input=generic_mlir,
            options=["--emit-function-info", "--allow-unregistered-dialect"],
        )
        func_name = func_info_str.splitlines()[0]
        arg_names = func_info_str.splitlines()[1].split(", ")
        secret_args = list(map(int, func_info_str.splitlines()[2].split(", ")))
      except CLIError as e:
        raise CompilerError(
            f"Failed to get function info from MLIR string: {e.stderr}",
            "<unknown>",
        )
    # Run Shape Inference
    mlir_textual = heir_opt.run_binary(
        input=mlir_raw_textual,
        options=["--shape-inference", "--mlir-print-debuginfo"],
    )
    if debug:
      mlir_in_filepath = Path(workspace_dir) / f"{func_name}.in.mlir"
      DebugMessage(
          f"Writing input MLIR w/ shape inference to {mlir_in_filepath}"
      )
      with open(mlir_in_filepath, "w") as f:
        f.write(mlir_textual)

    # Print type annotated version of the input
    if debug:
      mlirpath = Path(workspace_dir) / f"{func_name}.annotated.mlir"
      graphpath = Path(workspace_dir) / f"{func_name}.annotated.dot"
      heir_opt_output, graph = heir_opt.run_binary_stderr(
          input=mlir_textual,
          options=["--annotate-secretness", "--view-op-graph"],
      )
      DebugMessage(f"Writing secretness-annotated MLIR to {mlirpath}")
      with open(mlirpath, "w") as f:
        f.write(heir_opt_output)

      DebugMessage(f"Writing secretness-annotated graph to {graphpath}")
      with open(graphpath, "w") as f:
        f.write(graph)

    # Run heir_opt
    heir_opt_options.append(
        "--mlir-print-debuginfo"
    )  # to preserve location info
    if debug:
      heir_opt_options.append("--view-op-graph")
      DebugMessage(f"Running heir-opt {' '.join(heir_opt_options)}")
    try:
      heir_opt_output, graph = heir_opt.run_binary_stderr(
          input=mlir_textual, options=heir_opt_options
      )
    except CLIError as e:
      raise CompilerError(
          f"HEIR compilation for function {func_name} failed:\n"
          "running `heir-opt "
          f"{' '.join([str(x) for x in e.options])}` produced these errors:\n"
          f"{e.stderr}\n",
          numba_ir.loc if numba_ir else "<unknown>",
      )

    if debug:
      # Print output after heir_opt:
      mlirpath = Path(workspace_dir) / f"{func_name}.out.mlir"
      graphpath = Path(workspace_dir) / f"{func_name}.out.dot"
      DebugMessage(f"Writing output MLIR to {mlirpath}")
      with open(mlirpath, "w") as f:
        f.write(heir_opt_output)
      DebugMessage(f"Writing output graph to {graphpath}")
      with open(graphpath, "w") as f:
        f.write(graph)
    if not isinstance(backend, BackendInterface):
      raise TypeError(
          f"Expected BackendInterface instance but found: {type(backend)}If"
          " {type(backend)} is callable, maybe you forgot to instantiate it?"
      )

    # Run backend (which will call heir_translate and other tools, e.g., clang, as needed)
    if any(
        opt.startswith("--mlir-to-cggi") for opt in heir_opt_options
    ) and not isinstance(backend, CleartextBackend):
      raise NotImplementedError(
          "Backend compilation is unsupported for CGGI scheme, check CGGI"
          f" output at {mlirpath}"
      )

    result = backend.run_backend(
        workspace_dir,
        heir_opt,
        heir_translate,
        func_name,
        arg_names,
        secret_args,
        heir_opt_output,
        debug,
    )

    # Attach the original python func
    if isinstance(function_or_mlir_str, Callable):
      result.func = function_or_mlir_str  # type: ignore
    else:
      # If we got a string, we don't have the original function, so we create a suitable dummy
      # function that will raise an error if called.
      def dummy_func(*args, **kwargs):
        raise NotImplementedError(
            "This function was compiled from MLIR input, not a Python function."
            " Please use the compiled function directly."
        )

      result.func = dummy_func  # type: ignore

    return result

  finally:
    if debug:
      DebugMessage(
          f"Leaving workspace_dir {workspace_dir} for manual inspection.\n"
      )
    else:
      shutil.rmtree(workspace_dir)


def compile(
    mlir_str: Optional[str] = None,
    scheme: Optional[str] = "bgv",
    backend: Optional[BackendInterface] = None,
    config: Optional[heir_cli_config.HEIRConfig] = None,
    debug: Optional[bool] = False,
    heir_opt_options: Optional[list[str]] = None,
    # We intentionally break type inference here by describing the return type as Any
    # rather than the more reasonable Callable, as the latter will lead to, e.g.,
    # pyright enforcing the function's type annotation, e.g., with Secret[...]
    # and producing type checking errors whenever we call it with cleartext values
    # (e.g. standard Python int) or with Ctxt/Ptxt values with their own types.
) -> Any:
  """Compile a function to its private equivalent in FHE.

  Can be used either as a decorator on a python function (in which case mlir_str
  should NOT be set ) or directly with a string containing MLIR (mlir_str).

  Args:
      scheme: a string indicating the scheme to use. Options: 'bgv' (default),
        'ckks'.
      backend: backend object (e.g. OpenFHEBackend) to use for running the
        function
      config: a config object to control paths to the tools in the
        HEIRcompilation toolchain.
      debug: a boolean indicating whether to print debug information. Defaults
        to false.
      heir_opt_options: a list of strings to pass to the HEIR compiler as
        options. Defaults to None. If set, the `scheme` parameter is ignored.

  Returns:
    The decorator to apply to the given function.
  """
  if not config:
    if is_pip_installed():
      config = heir_cli_config.from_pip_installation()
    else:
      config = heir_cli_config.from_os_env()

  if not backend:
    if is_pip_installed():
      backend = OpenFHEBackend(openfhe_config.from_pip_installation())
    else:
      backend = OpenFHEBackend(openfhe_config.from_os_env())

  if debug and heir_opt_options is not None:
    DebugMessage(f"Overriding scheme with options {heir_opt_options}")

  # Decorator for Python Functions
  def decorator(func):
    try:
      return run_pipeline(
          func,
          heir_opt_options=heir_opt_options
          or ["--canonicalize", f"--mlir-to-{scheme}"],
          backend=backend,
          heir_config=config,
          debug=debug or False,
      )
    # If there is an error in the input program, the internal traceback is not going to be relevant to the user
    # So we catch the error, print it and then raise a SystemExit to stop the program
    except CompilerError as e:
      print(e)
      raise SystemExit(1)

  if not mlir_str:
    return decorator

  # Run on mlir_str directly
  try:
    return run_pipeline(
        mlir_str,
        heir_opt_options=heir_opt_options
        or ["--canonicalize", f"--mlir-to-{scheme}"],
        backend=backend,
        heir_config=config,
        debug=debug or False,
    )
  except CompilerError as e:
    print(e)
    raise SystemExit(1)
