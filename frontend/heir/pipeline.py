"""The compilation pipeline."""

import pathlib
import shutil
import tempfile
from typing import Any, Optional

from colorama import Fore, Style, init
from numba.core import compiler
from numba.core import sigutils
from numba.core.registry import cpu_target
from numba.core.typed_passes import type_inference_stage

from heir.backends.cleartext import CleartextBackend
from heir.backends.openfhe import OpenFHEBackend, config as openfhe_config
from heir.heir_cli import heir_cli, heir_cli_config
from heir.interfaces import BackendInterface, ClientInterface
from heir.mlir.types import parse_annotations
from heir.mlir_emitter import TextualMlirEmitter


# TODO (#1162): Allow for multiple functions in the same compilation. This requires switching from a decorator to a context thingy (`with heir.compile(...):`)

Path = pathlib.Path
HEIRConfig = heir_cli_config.HEIRConfig


def run_pipeline(
    function,
    heir_opt_options: list[str],
    backend: BackendInterface,
    heir_config: Optional[HEIRConfig] = None,
    debug: bool = False,
) -> ClientInterface:
  """Run the pipeline."""
  if not heir_config:
    heir_config = heir_cli_config.from_os_env()

  # The temporary workspace dir is so that heir-opt and the backend
  # can have places to write their output files. It is cleaned up once
  # the function returns, at which point the compiled python module has been
  # loaded into memory and the raw files are not needed.
  workspace_dir = tempfile.mkdtemp()
  try:
    ssa_ir = compiler.run_frontend(function)
    func_name = ssa_ir.func_id.func_name
    arg_names = ssa_ir.func_id.arg_names

    # Initialize Colorama for error and debug messages
    init(autoreset=True)

    # (Numba) Type Inference
    numba_signature = ""
    secret_args = ""
    try:
      numba_signature, secret_args, rettype = parse_annotations(
          function.__annotations__
      )
    except Exception as e:
      print(
          Fore.RED
          + Style.BRIGHT
          + "HEIR Error: Signature parsing failed for function"
          f" {func_name} with {type(e).__name__}: {e}"
      )
      raise
    try:
      fn_args, _ = sigutils.normalize_signature(numba_signature)
    except Exception as e:
      print(
          Fore.RED
          + Style.BRIGHT
          + "HEIR Error: Signature normalization failed for  function"
          f" {func_name} with signature {numba_signature} with"
          f" {type(e).__name__}: {e}"
      )
      raise
    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    typingctx.refresh()
    targetctx.refresh()
    try:
      typemap, restype, _, _ = type_inference_stage(
          typingctx, targetctx, ssa_ir, fn_args, None
      )
    except Exception as e:
      print(
          Fore.RED
          + Style.BRIGHT
          + f"HEIR Error: Type inference failed for function {func_name} with"
          f" signature {numba_signature} with {type(e).__name__}: {e}"
      )
      raise

    # If a result type was annotated, compare with numba
    if rettype is not None and debug:
      if rettype != restype:
        print(
            "HEIR Debug: Warning: user provided return type does not match"
            f" numba inference, expected {restype}, got {rettype}"
        )

    # Emit Textual IR
    mlir_textual = TextualMlirEmitter(
        ssa_ir, secret_args, typemap, restype
    ).emit()
    if debug:
      mlir_in_filepath = Path(workspace_dir) / f"{func_name}.in.mlir"
      print(f"HEIR Debug: Writing input MLIR to \t \t {mlir_in_filepath}")
      with open(mlir_in_filepath, "w") as f:
        f.write(mlir_textual)

    # Try to find heir_opt and heir_translate
    heir_opt = heir_cli.HeirOptBackend(heir_config.heir_opt_path)
    heir_translate = heir_cli.HeirTranslateBackend(
        binary_path=heir_config.heir_translate_path
    )

    # Print type annotated version of the input
    if debug:
      mlirpath = Path(workspace_dir) / f"{func_name}.annotated.mlir"
      graphpath = Path(workspace_dir) / f"{func_name}.annotated.dot"
      heir_opt_output, graph = heir_opt.run_binary_stderr(
          input=mlir_textual,
          options=["--annotate-secretness", "--view-op-graph"],
      )
      print(f"HEIR Debug: Writing secretness-annotated MLIR to \t {mlirpath}")
      with open(mlirpath, "w") as f:
        f.write(heir_opt_output)

      print(f"HEIR Debug: Writing secretness-annotated graph to \t {graphpath}")
      with open(graphpath, "w") as f:
        f.write(graph)

    # Run heir_opt
    if debug:
      heir_opt_options.append("--view-op-graph")
      print(
          "HEIR Debug: "
          + Style.BRIGHT
          + f"Running heir-opt {' '.join(heir_opt_options)}"
      )
    heir_opt_output, graph = heir_opt.run_binary_stderr(
        input=mlir_textual,
        options=heir_opt_options,
    )
    if debug:
      # Print output after heir_opt:
      mlirpath = Path(workspace_dir) / f"{func_name}.out.mlir"
      graphpath = Path(workspace_dir) / f"{func_name}.out.dot"
      print(f"HEIR Debug: Writing output MLIR to \t \t {mlirpath}")
      with open(mlirpath, "w") as f:
        f.write(heir_opt_output)
      print(f"HEIR Debug: Writing output graph to \t \t {graphpath}")
      with open(graphpath, "w") as f:
        f.write(graph)

    # Run backend (which will call heir_translate and other tools, e.g., clang, as needed)
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
    result.func = function  # type: ignore

    return result

  finally:
    if debug:
      print(
          f"HEIR Debug: Leaving workspace_dir {workspace_dir} for manual"
          " inspection.\n"
      )
    else:
      shutil.rmtree(workspace_dir)


def compile(
    scheme: Optional[str] = "bgv",
    backend: Optional[BackendInterface] = OpenFHEBackend(
        openfhe_config.from_os_env(),
    ),
    config: Optional[
        heir_cli_config.HEIRConfig
    ] = heir_cli_config.from_os_env(),
    debug: Optional[bool] = False,
    heir_opt_options: Optional[list[str]] = None,
) -> Any:
  # We intentionally break type inference here by describing the return type as Any
  # rather than the more reasonable Callable, as the latter will lead to, e.g.,
  # pyright enforcing the function's type annotation, e.g., with Secret[...]
  # and producing type checking errors whenever we call it with cleartext values
  # (e.g. standard Python int) or with Ctxt/Ptxt values with their own types.

  """Compile a function to its private equivalent in FHE.

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
  if debug and heir_opt_options is not None:
    print(f"HEIR Debug: Overriding scheme with options {heir_opt_options}")

  def decorator(func):
    return run_pipeline(
        func,
        heir_opt_options=heir_opt_options
        or ["--canonicalize", f"--mlir-to-{scheme}"],
        backend=backend or CleartextBackend(),
        heir_config=config or heir_cli_config.from_os_env(),
        debug=debug or False,
    )

  return decorator
