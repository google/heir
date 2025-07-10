"""The compilation pipeline."""

import os
import pathlib
import shutil
import tempfile
from typing import Any, Optional


from numba.core import compiler
from numba.core import sigutils
from numba.core.registry import cpu_target
from numba.core.typed_passes import type_inference_stage
from numba.core.types.misc import NoneType
from numba.core.types import Type as NumbaType
from numba.core.ir import FunctionIR

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


def run_pipeline(
    function,
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
    ssa_ir = compiler.run_frontend(function)
    if not isinstance(ssa_ir, FunctionIR):
      raise InternalCompilerError(
          f"Expected FunctionIR from numba frontend but got {type(ssa_ir)}"
      )
    func_name = ssa_ir.func_id.func_name
    arg_names = ssa_ir.func_id.arg_names

    # (Numba) Type Inference
    fn_args: list[NumbaType] = []
    secret_args: list[int] = []
    try:
      fn_args, secret_args, rettype = parse_annotations(
          function.__annotations__
      )
    except Exception as e:
      raise CompilerError(
          f"Signature parsing failed for function {func_name} with"
          f" {type(e).__name__}: {e}",
          ssa_ir.loc,
      )
    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    typingctx.refresh()
    targetctx.refresh()
    try:
      typemap, restype, _, _ = type_inference_stage(
          typingctx, targetctx, ssa_ir, fn_args, None
      )
    except Exception as e:
      raise CompilerError(
          f"Type inference failed for function {func_name} "
          f"with {type(e).__name__}: {e}",
          ssa_ir.loc,
      )

    # Check if we found a return type:
    if restype is None or isinstance(restype, NoneType):
      raise CompilerError(
          f"Type inference failed for function {func_name}: "
          "no return type could be determined.",
          ssa_ir.loc,
      )

    # If a result type was annotated, compare with numba
    if rettype is not None and debug:
      if rettype != restype:
        DebugMessage(
            " Warning: user provided return type does not match"
            f" numba inference, expected {restype}, got {rettype}",
            ssa_ir.loc,
        )

    # Emit Textual IR
    mlir_raw_textual = TextualMlirEmitter(
        ssa_ir, secret_args, typemap, restype
    ).emit()
    if debug:
      mlir_raw_filepath = Path(workspace_dir) / f"{func_name}.raw.mlir"
      DebugMessage(f"Writing raw input MLIR to {mlir_raw_filepath}")
      with open(mlir_raw_filepath, "w") as f:
        f.write(mlir_raw_textual)

    # Try to find heir_opt and heir_translate
    heir_opt = heir_cli.HeirOptBackend(heir_config.heir_opt_path)
    heir_translate = heir_cli.HeirTranslateBackend(
        binary_path=heir_config.heir_translate_path
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
          ssa_ir.loc,
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
    result.func = function  # type: ignore

    return result

  finally:
    if debug:
      DebugMessage(
          f"Leaving workspace_dir {workspace_dir} for manual inspection.\n"
      )
    else:
      shutil.rmtree(workspace_dir)


def _run_mlir_pipeline(
    mlir_raw_textual: str,
    func_name: str,
    arg_names: list[str],
    secret_args: list[int],
    heir_opt_options: list[str],
    backend: BackendInterface,
    config: HEIRConfig,
    debug: bool,
    workspace_dir: str,
    orig_func: Optional[Any] = None,
) -> ClientInterface:
  """Internal helper for MLIR-based pipelines."""
  heir_opt = heir_cli.HeirOptBackend(config.heir_opt_path)
  heir_translate = heir_cli.HeirTranslateBackend(
      binary_path=config.heir_translate_path
  )

  # Shape inference
  mlir_textual = heir_opt.run_binary(
      input=mlir_raw_textual,
      options=["--shape-inference", "--mlir-print-debuginfo"],
  )
  if debug:
    in_path = Path(workspace_dir) / f"{func_name}.in.mlir"
    DebugMessage(f"Writing input MLIR w/ shape inference to {in_path}")
    with open(in_path, "w") as f:
      f.write(mlir_textual)

  # Annotate secretness and graph
  if debug:
    annotated_path = Path(workspace_dir) / f"{func_name}.annotated.mlir"
    graph_path = Path(workspace_dir) / f"{func_name}.annotated.dot"
    heir_opt_output, graph = heir_opt.run_binary_stderr(
        input=mlir_textual, options=["--annotate-secretness", "--view-op-graph"]
    )
    DebugMessage(f"Writing secretness-annotated MLIR to {annotated_path}")
    with open(annotated_path, "w") as f:
      f.write(heir_opt_output)
    DebugMessage(f"Writing secretness-annotated graph to {graph_path}")
    with open(graph_path, "w") as f:
      f.write(graph)

  # Run heir-opt
  heir_opt_options.append("--mlir-print-debuginfo")
  if debug:
    heir_opt_options.append("--view-op-graph")
    DebugMessage(f"Running heir-opt {' '.join(heir_opt_options)}")
  try:
    heir_opt_output, graph = heir_opt.run_binary_stderr(
        input=mlir_textual, options=heir_opt_options
    )
  except CLIError as e:
    raise CompilerError(
        f"HEIR compilation for MLIR {func_name} failed:\nrunning `heir-opt"
        f" {' '.join([str(x) for x in e.options])}` produced these"
        f" errors:\n{e.stderr}\n",
        func_name,
    )

  if debug:
    out_mlir = Path(workspace_dir) / f"{func_name}.out.mlir"
    out_dot = Path(workspace_dir) / f"{func_name}.out.dot"
    DebugMessage(f"Writing output MLIR to {out_mlir}")
    with open(out_mlir, "w") as f:
      f.write(heir_opt_output)
    DebugMessage(f"Writing output graph to {out_dot}")
    with open(out_dot, "w") as f:
      f.write(graph)

  if not isinstance(backend, BackendInterface):
    raise TypeError(
        f"Expected BackendInterface instance but found: {type(backend)}. "
        "If it's callable, maybe you forgot to instantiate it?"
    )

  if any(
      opt.startswith("--mlir-to-cggi") for opt in heir_opt_options
  ) and not isinstance(backend, CleartextBackend):
    raise NotImplementedError(
        "Backend compilation is unsupported for CGGI scheme, check output MLIR"
        " files"
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
  result.func = orig_func
  return result


def compile(
    scheme: Optional[str] = "bgv",
    backend: Optional[BackendInterface] = None,
    config: Optional[heir_cli_config.HEIRConfig] = None,
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

  return decorator


def compile_mlir(
    mlir_raw_textual: str,
    func_name: str,
    arg_names: list[str],
    secret_args: list[int],
    scheme: Optional[str] = "bgv",
    backend: Optional[BackendInterface] = None,
    config: Optional[HEIRConfig] = None,
    debug: Optional[bool] = False,
    heir_opt_options: Optional[list[str]] = None,
) -> ClientInterface:
  """Compile an MLIR program to its private equivalent in FHE.

  Args:
    mlir_raw_textual: a string containing the MLIR program.
    func_name: the name of the function in the MLIR program to compile.
    arg_names: list of argument names in order.
    secret_args: list of indices of secret arguments.
    scheme: MLIR pipeline scheme, e.g. 'bgv' or 'ckks'. Ignored if heir_opt_options set.
    backend: backend to use for running the function.
    config: config object for HEIR toolchain paths.
    debug: whether to emit debug files.
    heir_opt_options: options to pass to heir-opt. If set, scheme is ignored.

  Returns:
    A ClientInterface for invoking the compiled function.
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

  options = heir_opt_options or ["--canonicalize", f"--mlir-to-{scheme}"]
  workspace_dir = tempfile.mkdtemp()
  try:
    # Write raw MLIR
    if debug:
      raw_path = Path(workspace_dir) / f"{func_name}.raw.mlir"
      DebugMessage(f"Writing raw input MLIR to {raw_path}")
      with open(raw_path, "w") as f:
        f.write(mlir_raw_textual)

    result = _run_mlir_pipeline(
        mlir_raw_textual,
        func_name,
        arg_names,
        secret_args,
        options,
        backend,
        config,
        debug,
        workspace_dir,
    )
    return result
  finally:
    if debug:
      DebugMessage(
          f"Leaving workspace_dir {workspace_dir} for manual inspection.\n"
      )
    else:
      shutil.rmtree(workspace_dir)
