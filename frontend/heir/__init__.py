# Necessary to adapt Numba's type inference behavior
from importlib import abc, util
import sys


class NumbaBuiltinOverrideFinder(abc.MetaPathFinder):

  def find_spec(self, fullname, path, target=None):
    if fullname == "numba.core.typing.old_builtins":
      return util.spec_from_file_location(
          fullname,
          # reverter files are in a sibling directory to this __init__.py file
          __file__.rsplit("/", 1)[0] + "/numba_nbep1_reverter/old_builtins.py",
      )
    elif fullname == "numba.core.typing.typeof":
      return util.spec_from_file_location(
          fullname,
          # reverter files are in a sibling directory to this __init__.py file
          __file__.rsplit("/", 1)[0] + "/numba_nbep1_reverter/typeof.py",
      )
    return None


# Insert our import hook
sys.meta_path.insert(0, NumbaBuiltinOverrideFinder())


## Normal __init__.py stuff below

from .pipeline import compile

__all__ = ["compile"]
