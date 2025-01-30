from abc import ABC, abstractmethod
from dataclasses import dataclass

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

class ClientInterface(ABC):

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

  def __call__(self, *args, **kwargs):
    """Forwards to the original function."""
    return self.func(*args, **kwargs)

class BackendInterface(ABC):
  ...
