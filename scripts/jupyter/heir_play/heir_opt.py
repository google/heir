import os
import shlex
import subprocess

from IPython.core.magic import Magics
from IPython.core.magic import cell_magic
from IPython.core.magic import magics_class


class BinaryMagic(Magics):
    def __init__(self, shell, binary_path):
        """
        Initialize a magic for running a binary with a given path.
        """
        super(BinaryMagic, self).__init__(shell)
        self.binary_path = binary_path

    def run_binary(self, line, cell):
        """
        Run the binary on the input cell.

        Args:
            line: The options to pass to the binary.
            cell: The input to pass to the binary.
        """
        print(f"Running {self.binary_path}...")
        completed_process = subprocess.run(
            [os.path.abspath(self.binary_path)] + shlex.split(line),
            input=cell,
            text=True,
            capture_output=True,
        )
        if completed_process.returncode != 0:
            print(f"Error running {self.binary_path}")
            print(completed_process.stdout)
            print(completed_process.stderr)
            return
        output = completed_process.stdout
        print(output)


@magics_class
class HeirOptMagic(BinaryMagic):
    def __init__(self, shell, binary_path="heir-opt"):
        """
        Initialize heir-opt with a path to the heir-opt binary.
        If not specified, will assume heir-opt is on the path.
        """
        super(HeirOptMagic, self).__init__(shell, binary_path)

    @cell_magic
    def heir_opt(self, line, cell):
        """
        Run heir-opt on the input cell.

        Args:
            line: The options to pass to heir-opt.
            cell: The input to pass to heir-opt.
        """
        return self.run_binary(line, cell)


@magics_class
class HeirTranslateMagic(BinaryMagic):
    def __init__(self, shell, binary_path="heir-translate"):
        """
        Initialize heir-translate with a path to the heir-translate binary.
        If not specified, will assume heir-translate is on the path.
        """
        super(HeirTranslateMagic, self).__init__(shell, binary_path)

    @cell_magic
    def heir_translate(self, line, cell):
        """
        Run heir-translate on the input cell.

        Args:
            line: The options to pass to heir-translate.
            cell: The input to pass to heir-translate.
        """
        return self.run_binary(line, cell)
