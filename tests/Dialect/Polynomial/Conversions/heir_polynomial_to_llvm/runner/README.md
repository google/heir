The tests in this directory test lowerings of polynomial-to-mod-arith by
lowering the result to machine code via LLVM and running them on hard-coded
inputs. To reduce the chance of the tests being specified incorrectly and the
burden of generating expected test outputs, the tests are automatically
generated.

`lower_mul_tests.toml` contains test specifications for `polynomial.mul`, and
each test has the following syntax

```
[[test]]
ideal = str (a sympy-parsable polynomial expression with variable `x`)
cmod = int
p0 = str (sympy-parsable polynomial in `x`)
p1 = str (sympy-parsable polynomial in `x`)
container_type = str (string representation of an mlir integer type)
```

This test then represents the correctness of the product of `p0` and `p1` in the
ring of polynomials mod `cmod` and `ideal`, using an underlying integer
container type of `container_type`.

For example,

```
[[test]]
ideal = "1 + x**12"
cmod = 4294967296
p0 = "1 + x**10"
p1 = "1 + x**11"
cmod_type = "i64"
coefficient_type = "i32"
```

Creates a test equivalent to

```
#ideal = #polynomial.int_polynomial<1 + x**12>
#ring = #polynomial.ring<coefficientType=i32 coefficientModulus=4294967296:i64, polynomialModulus=#ideal>
!poly_ty = !polynomial.polynomial<ring=#ring>

func.func @test() {
  %0 = polynomial.constant int<1 + x**10> : !poly_ty
  %1 = polynomial.constant int<1 + x**11> : !poly_ty
  %2 = polynomial.mul %0, %1 : !poly_ty

  %tensor = polynomial.to_tensor %2 : !poly_ty -> tensor<12xi32>
  %ref = bufferization.to_buffer %tensor : tensor<12xi32> to memref<12xi32>
  %U = memref.cast %ref : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
```

The script `generate_test_cases.py` reads this file in, parses the polynomials
using the [`sympy`](https://www.sympy.org/en/index.html) Python package,
computes the expected output, and prints an mlir test to a file as well as a
`cc_test` with the expected assertions.

After adding or updating `lower_mul_tests.toml`, re-generate the tests in this
directory with:

```
rm lower_mul_*.mlir
bazel run generate_test_cases -- --tests_toml_path $PWD/lower_mul_tests.toml --output_test_stem=$PWD/lower_mul_ --output_build_file $PWD/BUILD
```

<!-- mdformat global-off -->
