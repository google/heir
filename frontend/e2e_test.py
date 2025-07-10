from heir import compile, compile_mlir
from heir.mlir import F32, I16, I64, Secret
from heir.backends.cleartext import CleartextBackend


from absl.testing import absltest  # fmt: skip
class EndToEndTest(absltest.TestCase):

  def test_simple_arithmetic(self):

    @compile()  # defaults to BGV and OpenFHE
    def foo(a: Secret[I16], b: Secret[I16]):
      return a * a - b * b

    # Test cleartext functionality
    self.assertEqual(-15, foo.original(7, 8))

    # Test FHE functionality
    foo.setup()
    enc_a = foo.encrypt_a(7)
    enc_b = foo.encrypt_b(8)
    result_enc = foo.eval(enc_a, enc_b)
    result = foo.decrypt_result(result_enc)
    self.assertEqual(-15, result)

  def test_simple_example(self):

    @compile()
    def func(x: Secret[I16], y: Secret[I16]):
      sum = x + y
      diff = x - y
      mul = x * y
      expression = sum * diff + mul
      deadcode = expression * mul
      return expression

    # Test cleartext functionality
    self.assertEqual(41, func.original(7, 8))

    # Test FHE functionality
    self.assertEqual(41, func(7, 8))

  def test_manual_example(self):

    @compile()
    def manual(x: Secret[I16], y: Secret[I16]):
      return (x + y) * (x - y) + (x * y)

    manual.setup()  # runs keygen/etc
    enc_x = manual.encrypt_x(7)
    enc_y = manual.encrypt_y(8)
    result_enc = manual.eval(enc_x, enc_y)
    result = manual.decrypt_result(result_enc)

    # Test cleartext functionality
    self.assertEqual(41, manual.original(7, 8))

    # Test FHE functionality
    self.assertEqual(41, result)

  def test_loop_example(self):

    @compile()
    def loop_test(a: Secret[I64]):
      """An example function with a static loop."""
      result = 2
      for i in range(3):
        result = a + result
      return result

    # Test cleartext functionality
    self.assertEqual(8, loop_test.original(2))

    # Test FHE functionality
    self.assertEqual(8, loop_test(2))

  def test_ckks_example(self):

    @compile(scheme="ckks")
    def bar(x: Secret[F32], y: Secret[F32]):
      return (x + y) * (x - y) + (x * y)

    # Test cleartext functionality
    self.assertAlmostEqual(0.41, bar.original(0.7, 0.8))

    # Test FHE functionality
    self.assertAlmostEqual(0.41, bar(0.7, 0.8))

  def test_ctxt_ptxt_example(self):

    @compile()
    def baz(x: Secret[I16], y: Secret[I16], z: I16):
      ptxt_mul = x * z
      ctxt_mul = x * x
      ctxt_mul2 = y * y
      add = ctxt_mul + ctxt_mul2
      return ptxt_mul + add

    # Test cleartext functionality
    self.assertEqual(127, baz.original(7, 8, 2))

    # Test FHE functionality
    self.assertEqual(127, baz(7, 8, 2))

  def test_custom_example(self):

    @compile(
        heir_opt_options=[
            "--mlir-to-secret-arithmetic",
            "--canonicalize",
            "--cse",
        ],
        backend=CleartextBackend(),  # just runs the python function when `custom(...)` is called
        debug=True,
    )
    def custom(x: Secret[I16], y: Secret[I16]):
      return (x + y) * (x - y) + (x * y)

    # Test cleartext functionality
    self.assertEqual(41, custom.original(7, 8))

    # Test cleartext functionality via CleartextBackend
    self.assertEqual(41, custom(7, 8))

    # There's unfortunately no way to test the MLIR output here

  def test_mlir_example(self):
    # MLIR-as-string compilation path
    mlir_src = """
func.func @myfunc() -> i32 {
  %c = arith.constant 42 : i32
  return %c : i32
}
"""
    client = compile_mlir(
        mlir_src,
        func_name="myfunc",
        arg_names=[],
        secret_args=[],
    )
    # Test cleartext functionality
    self.assertEqual(42, client.original())
    # Test FHE functionality
    self.assertEqual(42, client())


if __name__ == "__main__":
  absltest.main()
