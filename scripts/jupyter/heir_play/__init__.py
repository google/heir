"""A magic for running heir-opt nightly binary"""
__version__ = "0.0.1"

import os

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
    load_nightly("abc")
    load_nightly("techmap.v")
    os.environ['HEIR_YOSYS_SCRIPTS_DIR'] = os.getcwd()
    os.environ['HEIR_ABC_BINARY'] = os.getcwd() + "/abc"
