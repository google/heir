"""Run a verilog script with a given set of inputs."""

import argparse
import subprocess
import sys

VerilogArg = tuple[str, int]


YOSYS_PRELUDE = """read_verilog {verilog_module}
proc"""

EVAL_TEMPLATE = 'eval {setters}'
ARG_SET = '-set {arg_name} {arg_val}'


parser = argparse.ArgumentParser(
    description=(
        'Run a verilog script with a given set of inputs, and print the outputs'
        ' to stdout.'
    )
)
parser.add_argument(
    '--verilog_module',
    type=str,
    help='A filepath to a verilog module to run.',
)
parser.add_argument(
    '--yosys_path',
    type=str,
    # default assumes `yosys` is already in the system path, which is true in
    # testing when lit is configured properly.
    default='yosys',
    help='The path to the yosys executable',
)
parser.add_argument(
    '--input',
    type=str,
    action='append',
    help=(
        'A list of one or more key-value pairs of argument names and input '
        'values to set in the form arg0=val0;arg1=val1;... Note this requires '
        "quoting the arg value as --input='arg0=val0;arg1=val1'"
    ),
)


def parse_inout_pair(inout_pair: str) -> VerilogArg:
  name, value = inout_pair.split('=')
  try:
    value = int(value)
  except ValueError:
    print(
        'Input values must be integers, but for given input '
        f'`{name}` found value `{value}`'
    )
    sys.exit(1)
  return (name, value)


def intstring_to_bytes(s):
  byte_length = max(1, (int(s).bit_length() + 7) // 8)
  return int(s).to_bytes(byte_length, byteorder='big')


def bitstring_to_bytes(s):
  byte_length = max(1, (len(s) + 7) // 8)
  return int(s, 2).to_bytes(byte_length, byteorder='big')


def parse_result(line: str) -> bytes:
  """Parse a line of Yosys output to the corresponding numeric output."""
  # line is
  #   Eval result: \_out_ = 8'00000100.
  rhs = line.split('=')[-1].strip()

  # rhs is either
  #   8'00000100.
  # or
  #   112.
  split_bin = rhs.rstrip('.').split("'")
  if len(split_bin) != 2:
    return intstring_to_bytes(split_bin[0])

  # bin_num is
  #   00000100
  #
  # This method just returns bytes since we aren't in a position
  # to determine if those bytes should be interpreted as signed
  # or unsigned integers.
  return bitstring_to_bytes(split_bin[1])


def run_verilog(
    input_set: list[list[VerilogArg]], args: argparse.Namespace
) -> list[tuple[tuple[int, ...], bytes]]:
  """Run yosys on the input verilog, and capture the output.

  As an example, the constructed commdn may look like

  yosys -Q -T -p "read_verilog hello_world.v; proc; eval -set arg1 0"
  """
  # Verilog outputs are of the form
  #
  #   @t=108s: 7. Executing EVAL pass (evaluate the circuit given an input).
  #   Eval result: \_out_ = 8'00000100.
  #
  # So we need to parse the output, scan for lines like `Eval result`,
  # and convert the outputs to integers.
  script_lines = YOSYS_PRELUDE.format(verilog_module=args.verilog_module).split(
      '\n'
  )

  def format_arg(name: str, value: int) -> str:
    return ARG_SET.format(arg_name=name, arg_val=value)

  def format_argset(assignment) -> str:
    return ' '.join(
        format_arg(arg_name, arg_val) for (arg_name, arg_val) in assignment
    )

  script_lines.extend([
      EVAL_TEMPLATE.format(setters=format_argset(assignment))
      for assignment in input_set
  ])
  # -Q and -T silence the yosys header/footer.
  cmd = (args.yosys_path, '-Q', '-T', '-p', '; '.join(script_lines))
  try:
    p = subprocess.run(cmd, capture_output=True, check=True, text=True)
  except subprocess.CalledProcessError as e:
    print(f'Yosys call failed with status code {e.returncode}')
    print('\n' + '=' * 25 + 'Captured stdout' + '=' * 25)
    print(f'{e.stdout}')
    print('\n' + '=' * 25 + 'Captured stderr' + '=' * 25)
    print(f'{e.stderr}')
    sys.exit(e.returncode)
  eval_result_lines = [x for x in p.stdout.split('\n') if 'Eval result' in x]
  parsed_results = [parse_result(line) for line in eval_result_lines]
  inputs = [
      tuple(iopair[1] for iopair in assignments) for assignments in input_set
  ]
  return list(zip(inputs, parsed_results))


def main(args: argparse.Namespace) -> None:
  input_sets = []
  if args.input:
    for input_set in args.input:
      input_sets.append(
          [parse_inout_pair(inout_pair) for inout_pair in input_set.split(';')]
      )
  else:
    input_sets.append([])

  results = run_verilog(input_sets, args)
  output = ''
  for _, output_bytes in results:
    # be consistent with python repr format for hex bytes
    hex_bytes = ''.join('\\x{:02x}'.format(x) for x in output_bytes)
    output += f"b'{hex_bytes}'\n"

  print(output)


if __name__ == '__main__':
  main(parser.parse_args())
