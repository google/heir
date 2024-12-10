from heir_py import heir_config
from heir_py import openfhe_config
from heir_py.pipeline import run_compiler

from absl.testing import absltest  # fmt: skip


class PipelineTest(absltest.TestCase):
    def test_simple_arithmetic(self):
        def foo(a, b):
            return a * a - b * b

        _heir_foo = run_compiler(
            foo,
            openfhe_config=openfhe_config.from_os_env(),
            heir_config=heir_config.from_os_env(),
        )

        cc = _heir_foo.foo__generate_crypto_context()
        kp = cc.KeyGen()
        _heir_foo.foo__configure_crypto_context(cc, kp.secretKey)
        arg0_enc = _heir_foo.foo__encrypt__arg0(cc, 7, kp.publicKey)
        arg1_enc = _heir_foo.foo__encrypt__arg1(cc, 8, kp.publicKey)
        res_enc = _heir_foo.foo(cc, arg0_enc, arg1_enc)
        res = _heir_foo.foo__decrypt__result0(cc, res_enc, kp.secretKey)

        self.assertEqual(-15, res)


if __name__ == "__main__":
    absltest.main()
