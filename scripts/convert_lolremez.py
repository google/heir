"""Convert a lolremez output polynomial string to the form needed for HEIR."""

import fire
import sympy


def convert_lolremez(
    name: str, poly_str: str, use_odd_trick: bool = True
) -> str:
  """Convert a lolremez output string to coefficient form.

  Example poly_str input:

    x*((((-2.6e+2*x**2+7.4e+2)*x**2-7.9e+2)*x**2+3.8e+2)*x**2-8.6e+1)*x**2+8.8

  Args:
    name: the name of the function being approximated
    poly_str: the polynomial expression to evaluate
    use_odd_trick: if True, map x -> x^2 and multiply whole poly by x

  Returns:
    The C++ header representation of the coefficient form of the polynomial.
  """
  if not use_odd_trick:
    raise NotImplemented()

  x = sympy.symbols("x")
  expr = eval(poly_str)
  expr = expr.replace(x, x**2)
  expr = x * expr

  poly = expr.as_poly()
  # change to coefficient list in degree-increasing order, with zeroes included
  coeffs = list(reversed(poly.as_list()))
  comma_sep_body = ",\n".join([f"  {v:.16f}" for v in coeffs])

  return f"""
// Polynomial approximation for {name}
//
//   {poly}
//
// Generated via
//
//   python scripts/convert_lolremez.py \\
//     --name='{name}' \\
//     --poly_str='{poly_str}' \\
//     --use_odd_trick='{use_odd_trick}'
//
static constexpr double {name.upper()}_APPROX_COEFFICIENTS[{len(coeffs)}] = {{
  {comma_sep_body}
}};
"""


if __name__ == "__main__":
  fire.Fire(convert_lolremez)
