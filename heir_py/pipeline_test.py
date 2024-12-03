from heir_py.pipeline import run_compiler

from absl.testing import absltest  # fmt: skip

class PipelineTest(absltest.TestCase):
    def test_simple_arithmetic(self):
        def foo(a, b):
            return a * a - b * b

        result = run_compiler(foo)
        for x in result:
            print(x)
        self.assertTrue(False)

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


if __name__ == '__main__':
  absltest.main()
