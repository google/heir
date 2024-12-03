from heir_py.pipeline import run_compiler


def test_simple_arithmetic():
  def foo(a, b):
    return a * a - b * b

  result = run_compiler(foo)
  assert result == """func.func @foo(%a: i64, %b: i64) -> (i64) {
  ^bb0:
    %0 = arith.muli %a, %a : i64
    %1 = arith.muli %b, %b : i64
    %2 = arith.subi %0, %1 : i64
    func.return %2 : i64
}
"""


def test_branch():
  def foo(a, b):
    if a < b:
      return a
    else:
      return b - 1

  result = run_compiler(foo)
  assert result == """func.func @foo(%a: i64, %b: i64) -> (i64) {
  ^bb0:
    %0 = arith.cmpi slt, %a, %b : i64
    cf.cond_br %0, ^bb14, ^bb18
  ^bb14:
    func.return %a : i64
  ^bb18:
    %1 = arith.constant 1 : i64
    %2 = arith.subi %b, %1 : i64
    func.return %2 : i64
}
"""
