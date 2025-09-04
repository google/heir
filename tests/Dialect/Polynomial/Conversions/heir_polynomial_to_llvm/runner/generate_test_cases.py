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
void _mlir_ciface_test_{test_number}(StridedMemRefType<int32_t, 1> *result);
}}

TEST(LowerMulTest, Test{test_number}) {{
  StridedMemRefType<int32_t, 1> result;
  int32_t data[{degree}];
  result.data = data;
  result.offset = 0;
  result.sizes[0] = {degree};
  result.strides[0] = 1;
  _mlir_ciface_test_{test_number}(&result);
  ASSERT_EQ(result.sizes[0], {degree});
{assertions}

  free(result.data);
}}
"""


TEST_TEMPLATE_BEFORE_GEN_TENSOR_OP = """#ideal_{test_number} = #polynomial.int_polynomial<{ideal}>
!coeff_ty_{test_number} = !mod_arith.int<{cmod}:{cmod_type}>
#ring_{test_number} = #polynomial.ring<coefficientType=!coeff_ty_{test_number}, polynomialModulus=#ideal_{test_number}>
!poly_ty_{test_number} = !polynomial.polynomial<ring=#ring_{test_number}>

func.func public @test_{test_number}() -> memref<{degree}xi32> {{
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<{p0}> : !poly_ty_{test_number}
  %1 = polynomial.constant int<{p1}> : !poly_ty_{test_number}
  %2 = polynomial.mul %0, %1 : !poly_ty_{test_number}
"""

TEST_TEMPLATE_AFTER_GEN_TENSOR_OP = """
  %ref = bufferization.to_buffer %tensor : tensor<{degree}xi32> to memref<{degree}xi32>
  return %ref : memref<{degree}xi32>
}}
"""

# We need to compare the final output in i32s because that's the only type
# currently supported by the c interface. So we need to truncate or extend the
# bits.
GEN_TENSOR_OP_TEMPLATE_WITH_TRUNC = """  %3 = polynomial.to_tensor %2 : !poly_ty_{test_number} -> tensor<{degree}x!coeff_ty_{test_number}>
  %4 = mod_arith.extract %3 : tensor<{degree}x!coeff_ty_{test_number}> -> tensor<{degree}x{cmod_type}>
  %tensor = arith.{trunc_ext_op} %4 : tensor<{degree}x{cmod_type}> to tensor<{degree}xi32>
"""
GEN_TENSOR_OP_TEMPLATE = """  %3 = polynomial.to_tensor %2 : !poly_ty_{test_number} -> tensor<{degree}x!coeff_ty_{test_number}>
  %tensor = mod_arith.extract %3 : tensor<{degree}x!coeff_ty_{test_number}> -> tensor<{degree}x{cmod_type}>
"""

TEST_TEMPLATE = """{before}
{gen_tensor_op}
{after}"""


def make_test_template(
    container_bit_width: int, test_number: int, degree: int, cmod_type: str
):
  if container_bit_width != 32:
    trunc_ext_op = "trunci" if container_bit_width > 32 else "extsi"
    return TEST_TEMPLATE.format(
        before=TEST_TEMPLATE_BEFORE_GEN_TENSOR_OP,
        after=TEST_TEMPLATE_AFTER_GEN_TENSOR_OP,
        gen_tensor_op=GEN_TENSOR_OP_TEMPLATE_WITH_TRUNC.format(
            test_number=test_number,
            degree=degree,
            cmod_type=cmod_type,
            trunc_ext_op=trunc_ext_op,
        ),
    )
  else:
    return TEST_TEMPLATE.format(
        before=TEST_TEMPLATE_BEFORE_GEN_TENSOR_OP,
        after=TEST_TEMPLATE_AFTER_GEN_TENSOR_OP,
        gen_tensor_op=GEN_TENSOR_OP_TEMPLATE.format(
            test_number=test_number,
            degree=degree,
            cmod_type=cmod_type,
        ),
    )


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

    x = sympy.Symbol("x")
    parsed_ideal = parse_to_sympy(ideal, x, cmod)
    parsed_p0 = parse_to_sympy(p0, x, cmod)
    parsed_p1 = parse_to_sympy(p1, x, cmod)
    domain = parsed_p0.domain

    expected_remainder = sympy.rem(parsed_p0 * parsed_p1, parsed_ideal, x)
    print(
        f"{expected_remainder.domain} : ({p0}) * ({p1}) ="
        f" {expected_remainder.as_expr()} mod ({ideal})"
    )
    coeff_list_len = parsed_ideal.degree()
    expected_coeffs = list(reversed(expected_remainder.all_coeffs()))
    container_width = int(cmod_type[1:])

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
          make_test_template(
              container_width, i, parsed_ideal.degree(), cmod_type
          ).format(
              ideal=ideal,
              cmod=cmod,
              cmod_type=cmod_type,
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
