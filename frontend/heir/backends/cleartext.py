"""Cleartext Backend."""

import colorama

Fore = colorama.Fore
Style = colorama.Style

from heir.interfaces import BackendInterface, CompilationResult, ClientInterface, EncValue
from heir.backends.util import common


class CleartextClientInterface(ClientInterface):

  def __init__(self, compilation_result: CompilationResult):
    self.compilation_result = compilation_result

  def setup(self):
    print(
        "HEIR Warning (Cleartext Backend): "
        + Fore.YELLOW
        + Style.BRIGHT
        + f"{self.compilation_result.func_name}.setup(..) "
        "is a no-op in the Cleartext Backend."
    )

  def decrypt_result(self, result):
    print(
        "HEIR Warning (Cleartext Backend): "
        + Fore.YELLOW
        + Style.BRIGHT
        + f"{self.compilation_result.func_name}.decrypt(..) "
        "is a no-op in the Cleartext Backend."
    )
    return result

  def __getattr__(self, key):

    if key.startswith("encrypt_"):
      arg_name = key[len("encrypt_") :]

      def wrapper(arg):
        print(
            "HEIR Warning (Cleartext Backend): "
            + Fore.YELLOW
            + Style.BRIGHT
            + f"{self.compilation_result.func_name}.{key}(..) "
            "does not perform any encryption in the Cleartext Backend."
        )
        return EncValue(arg_name, arg)

      return wrapper

    raise AttributeError(f"Attribute {key} not found")

  def eval(self, *args, **kwargs):
    print(
        "HEIR Warning (Cleartext Backend): "
        + Fore.YELLOW
        + Style.BRIGHT
        + f"{self.compilation_result.func_name}.eval(..) simply forwards to "
        f"{self.compilation_result.func_name}.original(..) "
        "in the Cleartext Backend."
    )

    stripped_args, stripped_kwargs = (
        common.strip_and_verify_eval_arg_consistency(
            self.compilation_result, *args, **kwargs
        )
    )

    return self.func(*stripped_args, **stripped_kwargs)

  def __call__(self, *args, **kwargs):
    print(
        "HEIR Warning (Cleartext Backend): "
        + Fore.YELLOW
        + Style.BRIGHT
        + f"{self.compilation_result.func_name}(..) is the same as"
        f" {self.compilation_result.func_name}.original(..) "
        "in the Cleartext Backend."
    )
    return self.func(*args, **kwargs)


class CleartextBackend(BackendInterface):

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

    result = CompilationResult(
        module=None,
        func_name=func_name,
        secret_args=secret_args,
        arg_names=arg_names,
    )

    return CleartextClientInterface(result)
