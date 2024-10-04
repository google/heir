"""A magic for running heir-opt nightly binary"""
__version__ = "0.0.1"

from .heir_opt import HeirOptMagic
from .heir_opt import HeirTranslateMagic
from .utils import load_nightly


def load_ipython_extension(ipython):
    ipython.register_magics(
        HeirOptMagic(ipython, binary_path=str(load_nightly("heir-opt")))
    )
    ipython.register_magics(
        HeirTranslateMagic(ipython, binary_path=str(load_nightly("heir-translate")))
    )
