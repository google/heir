"""The decorator entry point for the frontend."""

from typing import Optional
from abc import ABC, abstractmethod

from heir.backend.openfhe import openfhe_config
from frontend.heir.core import heir_cli_config
from heir.core.pipeline import CompilationResult, run_compiler

class CompilationResultInterface(ABC):

  @abstractmethod
  def setup(self):
    """Configure the initial cryptosystem and this wrapper interface."""
    ...

  @abstractmethod
  def decrypt_result(self, result, **kwargs):
    ...

  @abstractmethod
  def __getattr__(self, key):
    """Invoke a function with a dynamically-generated name:

    - encrypt_{arg_name}
    - {func_name}
    """
    ...

  @abstractmethod
  def __call__(self, *args, **kwargs):
    """Forwards to the original function."""
    ...

class OpenfheClientInterface(CompilationResultInterface):

  def __init__(self, compilation_result: CompilationResult):
    self.compilation_result = compilation_result

  def setup(self):
    # TODO(#1119): Rethink the server/client split
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
    return self.compilation_result.result_dec_func(
        crypto_context or self.crypto_context,
        result,
        secret_key or self.keypair.secretKey,
    )

  def __getattr__(self, key):
    if key.startswith("encrypt_"):
      arg_name = key[len("encrypt_") :]
      fn = self.compilation_result.arg_enc_funcs[arg_name]

      def wrapper(arg, *, crypto_context=None, public_key=None):
        return fn(
            crypto_context or self.crypto_context,
            arg,
            public_key or self.keypair.publicKey,
        )

      return wrapper

    if key == self.compilation_result.func_name or key == "eval":
      fn = self.compilation_result.main_func

      def wrapper(*args, crypto_context=None):
        return fn(crypto_context or self.crypto_context, *args)

      return wrapper

    try:
      getattr(self.compilation_result.module, key)
    except AttributeError:
      raise AttributeError(f"Attribute {key} not found")

  def __call__(self, *args, **kwargs):
    self.setup()
    arg_names = list(self.compilation_result.arg_names)
    num_args = len(arg_names)

    if len(args) != num_args:
      raise ValueError(
          f"Expected {num_args} arguments, got {len(args)}. "
          f"Expected args: {arg_names}"
      )

    args_encrypted = [
      getattr(self, f"encrypt_{arg_name}")(arg) if i in self.compilation_result.secret_args else arg
      for i, (arg_name, arg) in enumerate(zip(arg_names, args))
    ]

    result_encrypted = getattr(self, self.compilation_result.func_name)(
        *args_encrypted
    )
    return self.decrypt_result(result_encrypted)


def compile(
    scheme: str = "bgv",
    backend: str = "openfhe",
    backend_config: Optional[openfhe_config.OpenFHEConfig] = openfhe_config.DEFAULT_INSTALLED_OPENFHE_CONFIG,
    heir_config: Optional[heir_cli_config.HEIRConfig] = heir_cli_config.DEVELOPMENT_HEIR_CONFIG,
    debug : Optional[bool] = False,
    heir_opt_options : Optional[list[str]] = None
):
  """Compile a function to its private equivalent in FHE.

  Args:
      scheme: a string indicating the scheme to use. Options: 'bgv' (default),
        'ckks'.
      backend: a string indicating the backend to use. Options: 'openfhe'
        (default).
      backend_config: a config object to control system-specific paths for the
        backend in question.
      heir_config: a config object to control paths to the tools in the HEIR
        compilation toolchain.
      debug: a boolean indicating whether to print debug information. Defaults to false.
      heir_opt_options: a list of strings to pass to the HEIR compiler as options. Defaults to None.
      If set, the `scheme` parameter is ignored.


  Returns:
    The decorator to apply to the given function.
  """

  def decorator(func):
    compilation_result = run_compiler(
        func,
        scheme,
        backend,
        openfhe_config=backend_config or openfhe_config.from_os_env(),
        heir_config=heir_config or heir_config.from_os_env(),
        debug = debug,
        heir_opt_options = heir_opt_options
    )
    if backend == "openfhe":
      return OpenfheClientInterface(compilation_result)
    elif backend =="heracles":
      # FIXME: Implement HeraclesClientInterface
      pass
    elif backend == None:
       pass
    else:
      raise ValueError(f"Unknown backend: {backend}")

  return decorator
