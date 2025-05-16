"""OpenFHE Backend."""

import importlib
import pathlib
import sys

import colorama

Fore = colorama.Fore
Style = colorama.Style

from heir.interfaces import (
    BackendInterface,
    CompilationResult,
    ClientInterface,
    EncValue,
)
from heir.backends.util import (
    cpp_compiler,
    pybind_helpers,
    common,
)
from .config import OpenFHEConfig
from functools import partial

Path = pathlib.Path
pyconfig_ext_suffix = pybind_helpers.pyconfig_ext_suffix
pybind11_includes = pybind_helpers.pybind11_includes
pybind11_libs = pybind_helpers.pybind11_libs
python_link_libs = pybind_helpers.python_link_libs


class OpenfheClientInterface(ClientInterface):

  def __init__(self, compilation_result: CompilationResult):
    self.compilation_result = compilation_result

  def setup(self):
    # TODO (#1119): Rethink the server/client split
    # TODO (#1162) : Fix "ImportError: generic_type: type "PublicKey" is already registered!" when doing setup twice. (Required to allow multiple compilations in same python file)

    if self.compilation_result.setup_funcs is None:
      raise ValueError("No setup functions found in compilation result")

    self.crypto_context = self.compilation_result.setup_funcs[
        "generate_crypto_context"
    ]()
    self.keypair = self.crypto_context.KeyGen()
    # Really, "configure" is just setting up non-public key material, and
    # all public parameter configuration is done on
    # generate_crypto_context.
    self.compilation_result.setup_funcs["configure_crypto_context"](
        self.crypto_context, self.keypair.secretKey
    )

  def decrypt_result(self, result, *, crypto_context=None, secret_key=None):
    if self.compilation_result.result_dec_func is None:
      raise ValueError("No decryption function found in compilation result")

    return self.compilation_result.result_dec_func(
        crypto_context or self.crypto_context,
        result,
        secret_key or self.keypair.secretKey,
    )

  def __getattr__(self, key):
    if key == "crypto_context":
      msg = (
          f"HEIR Error: Please call {self.compilation_result.func_name}.setup()"
          " before calling"
          f" {self.compilation_result.func_name}.encrypt/eval/decrypt"
      )
      colorama.init(autoreset=True)
      print(Fore.RED + Style.BRIGHT + msg)
      raise RuntimeError(msg)

    if key.startswith("encrypt_"):
      # TODO (#1162): Prevent people from doing a = enc_x, b = enc_y for foo(x,y) but then calling foo(b,a)!
      # TODO (#1119): expose ctxt serialization in python
      if self.compilation_result.arg_enc_funcs is None:
        raise ValueError("No encryption functions found in compilation result")

      arg_name = key[len("encrypt_") :]
      enc_fn = self.compilation_result.arg_enc_funcs[arg_name]

      def enc_wrapper(arg, *, crypto_context=None, public_key=None):
        return enc_fn(
            crypto_context or self.crypto_context,
            arg,
            public_key or self.keypair.publicKey,
        )

      return enc_wrapper

    try:
      return getattr(self.compilation_result.module, key)
    except AttributeError:
      raise AttributeError(f"Attribute {key} not found")

  def eval(self, *args, **kwargs):
    # Check that the arguments provided are consistent:
    stripped_args, stripped_kwargs = (
        common.strip_and_verify_eval_arg_consistency(
            self.compilation_result, *args, **kwargs
        )
    )

    # Get underlying eval func
    fn = self.compilation_result.main_func
    if fn is None:
      raise ValueError("No main function found in compilation result")

    return fn(self.crypto_context, *stripped_args, **stripped_kwargs)

  def __call__(self, *args, **kwargs):
    # Setup
    self.setup()

    # Encrypt
    if self.compilation_result.arg_enc_funcs is None:
      raise ValueError("No encryption functions found in compilation result")
    if len(self.compilation_result.arg_names) != len(args):
      raise ValueError(
          f"Expected {len(self.compilation_result.arg_names)} arguments,"
          f" got {len(args)}"
      )

    new_args = []
    for i, arg in enumerate(args):
      if i in self.compilation_result.secret_args:
        arg_name = self.compilation_result.arg_names[i]
        enc_fn = self.compilation_result.arg_enc_funcs[arg_name]
        new_args.append(
            enc_fn(self.crypto_context, arg, self.keypair.publicKey)
        )
      else:
        new_args.append(arg)

    new_kw_args = {}
    for arg_name, arg in kwargs.items():
      i = self.compilation_result.arg_names.index(arg_name)
      if i in self.compilation_result.secret_args:
        enc_fn = self.compilation_result.arg_enc_funcs[arg_name]
        new_kw_args[arg_name] = enc_fn(
            self.crypto_context, arg, self.keypair.publicKey
        )
      else:
        new_kw_args[arg_name] = arg

    # Eval
    result = self.eval(*new_args, **new_kw_args)

    # Decrypt
    return self.decrypt_result(result)


class OpenFHEBackend(BackendInterface):

  def __init__(self, openfhe_config: OpenFHEConfig):
    self.openfhe_config = openfhe_config
    self.heir_opt_options = []

  def run_backend(
      self,
      workspace_dir,
      heir_opt,
      heir_translate,
      func_name,
      arg_names,
      secret_args,
      heir_opt_output,
      debug,
  ):
    colorama.init(autoreset=True)

    # Convert from "scheme" to openfhe:
    heir_opt_options = [
        f"--scheme-to-openfhe=entry-function={func_name}",
        "--mlir-print-debuginfo",
    ]
    if debug:
      heir_opt_options.append("--view-op-graph")
      print(
          "HEIRpy Debug (OpenFHE Backend): "
          + Style.BRIGHT
          + f"Running heir-opt {' '.join(heir_opt_options)}"
      )
    heir_opt_output, graph = heir_opt.run_binary_stderr(
        input=heir_opt_output,
        options=(heir_opt_options),
    )
    if debug:
      # Print output after heir_opt:
      mlirpath = Path(workspace_dir) / f"{func_name}.backend.mlir"
      graphpath = Path(workspace_dir) / f"{func_name}.backend.dot"
      print(
          f"HEIRpy Debug (OpenFHE Backend): Writing backend MLIR to {mlirpath}"
      )
      with open(mlirpath, "w") as f:
        f.write(heir_opt_output)
      print(
          "HEIRpy Debug (OpenFHE Backend): Writing backend graph to"
          f" {graphpath}"
      )
      with open(graphpath, "w") as f:
        f.write(graph)

    # Translate to *.cpp and Pybind
    module_name = f"_heir_{func_name}"
    cpp_filepath = Path(workspace_dir) / f"{func_name}.cpp"
    h_filepath = Path(workspace_dir) / f"{func_name}.h"
    pybind_filepath = Path(workspace_dir) / f"{func_name}_bindings.cpp"
    include_type_flag = (
        "--openfhe-include-type=" + self.openfhe_config.include_type
    )
    header_options = [
        "--emit-openfhe-pke-header",
        include_type_flag,
        "-o",
        h_filepath,
    ]
    cpp_options = ["--emit-openfhe-pke", include_type_flag, "-o", cpp_filepath]
    pybind_options = [
        "--emit-openfhe-pke-pybind",
        f"--pybind-header-include={h_filepath.name}",
        f"--pybind-module-name={module_name}",
        "-o",
        pybind_filepath,
    ]
    if debug:
      print(
          "HEIRpy Debug (OpenFHE Backend): "
          + Style.BRIGHT
          + f"Running heir-translate {' '.join(str(o) for o in header_options)}"
      )
    heir_translate.run_binary(
        input=heir_opt_output,
        options=header_options,
    )
    if debug:
      print(
          "HEIRpy Debug (OpenFHE Backend): "
          + Style.BRIGHT
          + f"Running heir-translate {' '.join(str(o) for o in cpp_options)}"
      )
    heir_translate.run_binary(
        input=heir_opt_output,
        options=cpp_options,
    )
    if debug:
      print(
          "HEIRpy Debug (OpenFHE Backend): "
          + Style.BRIGHT
          + f"Running heir-translate {' '.join(str(o) for o in pybind_options)}"
      )
    heir_translate.run_binary(
        input=heir_opt_output,
        options=pybind_options,
    )

    cpp_compiler_backend = cpp_compiler.CppCompilerBackend()
    so_filepath = Path(workspace_dir) / f"{func_name}.so"
    linker_search_paths = [self.openfhe_config.lib_dir]

    def debug_printer(args):
      print(
          "HEIRpy Debug (OpenFHE Backend): "
          + Style.BRIGHT
          + f"Running cpp compiler {' '.join(str(arg) for arg in args)}"
      )

    cpp_compiler_backend.compile_to_shared_object(
        cpp_source_filepath=cpp_filepath,
        shared_object_output_filepath=so_filepath,
        include_paths=self.openfhe_config.include_dirs,
        linker_search_paths=linker_search_paths,
        link_libs=self.openfhe_config.link_libs,
        arg_printer=debug_printer if debug else None,
    )

    ext_suffix = pyconfig_ext_suffix()
    pybind_so_filepath = Path(workspace_dir) / f"{module_name}{ext_suffix}"
    cpp_compiler_backend.compile_to_shared_object(
        cpp_source_filepath=pybind_filepath,
        shared_object_output_filepath=pybind_so_filepath,
        include_paths=self.openfhe_config.include_dirs
        + pybind11_includes()
        + [workspace_dir],
        linker_search_paths=linker_search_paths + pybind11_libs(),
        link_libs=self.openfhe_config.link_libs + python_link_libs(),
        linker_args=["-rpath", ":".join(linker_search_paths)],
        abs_link_lib_paths=[so_filepath],
        arg_printer=debug_printer if debug else None,
    )

    sys.path.append(workspace_dir)
    importlib.invalidate_caches()
    bound_module = importlib.import_module(module_name)

    def wrap_enc_fn(arg_name, enc_fn, *args, **kwargs):
      return EncValue(arg_name, enc_fn(*args, **kwargs))

    result = CompilationResult(
        module=bound_module,
        func_name=func_name,
        secret_args=secret_args,
        arg_names=arg_names,
        arg_enc_funcs={
            arg_name: partial(
                wrap_enc_fn,
                arg_name,
                getattr(bound_module, f"{func_name}__encrypt__arg{i}"),
            )
            for i, arg_name in enumerate(arg_names)
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

    return OpenfheClientInterface(result)
