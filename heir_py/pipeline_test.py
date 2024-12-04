from heir_py import openfhe_config
from heir_py.pipeline import run_compiler

from absl.testing import absltest  # fmt: skip


class PipelineTest(absltest.TestCase):
    def test_simple_arithmetic(self):
        def foo(a, b):
            return a * a - b * b

        _heir_foo = run_compiler(foo, openfhe_config=openfhe_config.from_os_env(debug=True))

        cc = _heir_foo.foo__generate_crypto_context()
        kp = cc.KeyGen()
        _heir_foo.foo__configure_crypto_context(cc, kp.secretKey)
        arg0_enc = _heir_foo.foo__encrypt__arg0(cc, 7, kp.publicKey)
        arg1_enc = _heir_foo.foo__encrypt__arg1(cc, 8, kp.publicKey)
        res_enc = _heir_foo.foo(cc, arg0_enc, arg1_enc)
        res = _heir_foo.foo__decrypt__result0(cc, res_enc, kp.secretKey)

        self.assertEqual(-15, res)

    # def test_branch(self):
    #     def foo(a, b):
    #         if a < b:
    #             return a
    #         else:
    #             return b - 1

    #     result = run_compiler(foo)
    #     self.assertEqual(
    #         result,
    #         """func.func @foo(%a: i64, %b: i64) -> (i64) {
    #   ^bb0:
    #     %0 = arith.cmpi slt, %a, %b : i64
    #     cf.cond_br %0, ^bb14, ^bb18
    #   ^bb14:
    #     func.return %a : i64
    #   ^bb18:
    #     %1 = arith.constant 1 : i64
    #     %2 = arith.subi %b, %1 : i64
    #     func.return %2 : i64
    # }
    # """,
    #     )


if __name__ == "__main__":
    absltest.main()
