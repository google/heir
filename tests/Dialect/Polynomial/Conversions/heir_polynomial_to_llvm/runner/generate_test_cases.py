"""A test case generator for lowering polynomial.mul."""

import argparse
import sys

import sympy
import tomli

MAIN_C_SRC_TEMPLATE = """#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"  // from @absltest
#include "tests/llvm_runner/memref_types.h"

extern "C" {{
void _mlir_ciface_test_{test_number}(StridedMemRefType<{capi_type}, 1> *result);
}}

TEST(LowerMulTest, Test{test_number}) {{
  StridedMemRefType<{capi_type}, 1> result;
  _mlir_ciface_test_{test_number}(&result);
  ASSERT_EQ(result.sizes[0], {degree});
{assertions}
  free(result.basePtr);
}}
"""


TEST_TEMPLATE = """#ideal_{test_number} = #polynomial.int_polynomial<{ideal}>
!coeff_ty_{test_number} = !mod_arith.int<{cmod}:{cmod_type}>
#ring_{test_number} = #polynomial.ring<coefficientType=!coeff_ty_{test_number}, polynomialModulus=#ideal_{test_number}>
!poly_ty_{test_number} = !polynomial.polynomial<ring=#ring_{test_number}>

func.func public @test_{test_number}() -> !poly_ty_{test_number} {{
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<{p0}> : !poly_ty_{test_number}
  %1 = polynomial.constant int<{p1}> : !poly_ty_{test_number}
  %2 = polynomial.mul %0, %1 : !poly_ty_{test_number}
  return %2 : !poly_ty_{test_number}
"""


parser = argparse.ArgumentParser(
    description=(
        "Generate a list of integration tests for lowering polynomial.mul ops"
    )
)
parser.add_argument(
    "--tests_toml_path",
    type=str,
    help="A filepath to a toml file containing a list of tests to generate.",
)
parser.add_argument(
    "--output_test_stem",
    type=str,
    help=(
        "If specified, generate each test as its own file with this argument"
        " as the path stem."
    ),
)
parser.add_argument(
    "--output_build_file",
    type=str,
    help="A filepath to the output BUILD file.",
)


def get_key_or_err(the_dict, key: str, error: str = None):
  error = error or f"[[test]] Missing key {key}, parsed dict={the_dict}"
  try:
    return the_dict[key]
  except KeyError:
    print(error)
    sys.exit(1)


def parse_polynomial(poly_str: str) -> list[tuple[int, int]]:
  """Parse a polynomial string into a list of coeff-degree pairs."""
  terms = [x.strip().split("x**") for x in poly_str.split("+")]
  term_dict = dict()
  for term in terms:
    term = [x.strip() for x in term]
    if term[0]:
      term_coeff = int(term[0])
    else:
      term_coeff = 1
    if len(term) == 1:
      term_dict[0] = term_coeff
    else:
      degree = int(term[1])
      term_dict[degree] = term_coeff

  return list((coeff, degree) for (degree, coeff) in term_dict.items())


def parse_to_sympy(poly_str: str, var: sympy.Symbol, cmod: int):
  terms = parse_polynomial(poly_str)
  poly = 0
  for coeff, degree in terms:
    poly += coeff * var**degree
  return poly.as_poly(domain=f"ZZ[{cmod}]")


def main(args: argparse.Namespace) -> None:
  if not args.tests_toml_path:
    print("No test config was passed via --tests_toml_path")
    sys.exit(1)

  if not args.output_test_stem:
    print("Must pass --output_test_stem")
    sys.exit(1)

  if not args.output_build_file:
    print("Must pass --output_build_file")
    sys.exit(1)

  with open(args.tests_toml_path, "rb") as infile:
    config = tomli.load(infile)

  tests = []
  try:
    tests = config["test"]
  except KeyError:
    print("TOML file must contain one or more sections like [[test]]")
    sys.exit(1)
  test_count = len(tests)

  print(f"Generating {test_count} tests...")
  build_targets = ""
  for i, test in enumerate(tests):
    (ideal, cmod, p0, p1, cmod_type) = (
        get_key_or_err(test, s)
        for s in ["ideal", "cmod", "p0", "p1", "cmod_type"]
    )
    capi_type = "int64_t" if cmod_type == "i64" else "int32_t"

    x = sympy.Symbol("x")
    parsed_ideal = parse_to_sympy(ideal, x, cmod)
    parsed_p0 = parse_to_sympy(p0, x, cmod)
    parsed_p1 = parse_to_sympy(p1, x, cmod)

    expected_remainder = sympy.rem(parsed_p0 * parsed_p1, parsed_ideal, x)
    print(
        f"{expected_remainder.domain} : ({p0}) * ({p1}) ="
        f" {expected_remainder.as_expr()} mod ({ideal})"
    )
    expected_coeffs = list(reversed(expected_remainder.all_coeffs()))

    # For whatever reason, sympy won't preserve the domain of the
    # coefficients after `rem`, so I have to manually convert any fractional
    # coefficients to their modular inverse equivalents.
    for j, exp_coeff in enumerate(expected_coeffs):
      p = exp_coeff.p
      q = exp_coeff.q
      q_inv = sympy.mod_inverse(q, cmod)
      result = (p * q_inv) % cmod
      expected_coeffs[j] = result

    assertions = []
    for j, coeff in enumerate(expected_coeffs):
      assertions.append(f"  EXPECT_EQ(result.data[{j}], (int32_t){coeff});")

    with open(f"{args.output_test_stem}{i}.mlir", "w") as outfile:
      outfile.write(
          TEST_TEMPLATE.format(
              ideal=ideal,
              cmod=cmod,
              cmod_type=cmod_type,
              capi_type=capi_type,
              p0=p0,
              p1=p1,
              test_number=i,
              degree=parsed_ideal.degree(),
          )
      )
    with open(f"{args.output_test_stem}{i}_test.cc", "w") as outfile:
      outfile.write(
          MAIN_C_SRC_TEMPLATE.format(
              test_number=i,
              degree=parsed_ideal.degree(),
              assertions="\n".join(assertions),
          )
      )
    build_targets += f"""\nllvm_runner_test(
    name = \"lower_mul_{i}\",
    heir_opt_flags = [
        "--emit-c-interface",
        "--heir-polynomial-to-llvm",
    ],
    main_c_src = \"lower_mul_{i}_test.cc\",
    mlir_src = \"lower_mul_{i}.mlir\",
    deps = [
        \"@absltest//:gtest_main\",
        \"@heir//tests/llvm_runner:memrefCopy\",
    ],
)
"""

  # Now, the build file content has some boilerplate, then a comment
  # `# -- AUTOGENERATED_BELOW`
  # then the autogenerated rules.
  marker = "# -- AUTOGENERATED_BELOW"
  with open(args.output_build_file, "r") as f:
    content = f.read()
    parts = content.split(marker)
    if len(parts) != 2:
      print(
          "Expected to find the marker"
          f" '{marker}' exactly once in {args.output_build_file}"
      )
      sys.exit(1)
    header = parts[0]

  with open(args.output_build_file, "w") as outfile:
    outfile.write(header)
    outfile.write(marker + "\n")
    outfile.write(build_targets)

  print("Done")


if __name__ == "__main__":
  main(parser.parse_args())
