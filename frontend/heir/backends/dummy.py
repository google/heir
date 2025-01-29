"""Dummy Backend."""

from colorama import Fore, Style, init

from heir.core import BackendInterface, CompilationResult,  ClientInterface

class DummyClientInterface(ClientInterface):

  def __init__(self, compilation_result: CompilationResult):
    self.compilation_result = compilation_result

  def setup(self):
    print("HEIR Warning (Dummy Backend): " + Fore.YELLOW + Style.BRIGHT + f"{self.compilation_result.func_name}.setup() is a no-op in the Dummy Backend")

  def decrypt_result(self, result):
    print("HEIR Warning (Dummy Backend): " + Fore.YELLOW + Style.BRIGHT + f"{self.compilation_result.func_name}.decrypt() is a no-op in the Dummy Backend")
    return result

  def __getattr__(self, key):

    if key.startswith("encrypt_"):
      arg_name = key[len("encrypt_") :]

      def wrapper(arg):
        print("HEIR Warning (Dummy Backend): " + Fore.YELLOW + Style.BRIGHT + f"{self.compilation_result.func_name}.{key}() is a no-op in the Dummy Backend")
        return arg

      return wrapper

    if key == self.compilation_result.func_name or key == "eval":
      print("HEIR Warning (Dummy Backend): " + Fore.YELLOW + Style.BRIGHT + f"{self.compilation_result.func_name}.eval() is the same as {self.compilation_result.func_name}() in the Dummy Backend.")
      return self.func

    raise AttributeError(f"Attribute {key} not found")


class DummyBackend(BackendInterface):

    def run_backend(self, workspace_dir, heir_opt, heir_translate, func_name, arg_names, secret_args, heir_opt_output, debug):

        result = CompilationResult(
              module=None,
              func_name=func_name,
              secret_args=secret_args,
              arg_names=arg_names,
              arg_enc_funcs=None,
              result_dec_func=None,
              main_func=None,
              setup_funcs=None
          )

        return DummyClientInterface(result)
