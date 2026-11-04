"""Smoke test for the prototype HEIR python bindings."""

import unittest

from heir_mlir import ir
from heir_mlir import passmanager
from heir_mlir.dialects import arith
from heir_mlir.dialects import func
from heir_mlir.dialects import secret


class HeirMlirTest(unittest.TestCase):

  def test_build_secret_ir_in_memory(self):
    with ir.Context(), ir.Location.unknown():
      module = ir.Module.create()
      i32 = ir.IntegerType.get_signless(32)
      secret_i32 = ir.Type.parse("!secret.secret<i32>")
      with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(i32)
        def conceal_add(x):
          c = arith.ConstantOp(i32, 7)
          y = arith.AddIOp(x, c)
          return secret.ConcealOp(secret_i32, y).result

      self.assertTrue(module.operation.verify())
      text = str(module)
      self.assertIn("secret.conceal", text)
      self.assertIn("!secret.secret<i32>", text)

  def test_run_heir_pass(self):
    with ir.Context():
      module = ir.Module.parse("""
        func.func @f(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
          %0 = secret.generic(%arg0: !secret.secret<i32>) {
          ^body(%x: i32):
            %1 = arith.addi %x, %x : i32
            secret.yield %1 : i32
          } -> !secret.secret<i32>
          return %0 : !secret.secret<i32>
        }
      """)
      pm = passmanager.PassManager.parse(
          "builtin.module(secret-forget-secrets,canonicalize)"
      )
      pm.run(module.operation)
      text = str(module)
      self.assertNotIn("secret", text)
      self.assertIn("arith.addi", text)


if __name__ == "__main__":
  unittest.main()
