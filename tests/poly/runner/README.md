To re-generate the tests in this directory:

```
cd tests/poly/runner
rm lower_mul_*.mlir
bazel run generate_test_cases -- --tests_toml_path $PWD/lower_mul_tests.toml --output_test_stem=$PWD/lower_mul_
```
