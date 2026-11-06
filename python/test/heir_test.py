"""Tests for the heir python package."""

import importlib
import unittest

import heir
from heir import ir
from heir import types
from heir.dialects import arith
from heir.dialects import func
from heir.dialects import secret

DOT_PRODUCT = """
func.func @dot_product(%arg0: tensor<8xi16> {secret.secret}, %arg1: tensor<8xi16> {secret.secret}) -> i16 {
  %cst = arith.constant dense<0> : tensor<i16>
  %0 = linalg.dot ins(%arg0, %arg1 : tensor<8xi16>, tensor<8xi16>) outs(%cst : tensor<i16>) -> tensor<i16>
  %1 = tensor.extract %0[] : tensor<i16>
  return %1 : i16
}
"""

HEIR_DIALECTS = [
    "bgv",
    "cggi",
    "cheddar",
    "ckks",
    "comb",
    "debug",
    "jaxite",
    "jaxiteword",
    "kernel",
    "key_mgmt",
    "lattigo",
    "lwe",
    "math_ext",
    "mgmt",
    "mod_arith",
    "openfhe",
    "orion",
    "polynomial",
    "poulpy",
    "preprocessing",
    "random",
    "rns",
    "scifrbool",
    "secret",
    "tensor_ext",
    "tfhe_rust",
    "tfhe_rust_bool",
]

UPSTREAM_DIALECTS = [
    "affine",
    "arith",
    "bufferization",
    "builtin",
    "emitc",
    "func",
    "math",
    "memref",
    "scf",
    "tensor",
]


class EndToEndTest(unittest.TestCase):

  def test_compile_to_openfhe(self):
    with heir.context():
      module = heir.parse(DOT_PRODUCT)
      heir.run(
          module,
          "--annotate-module=backend=openfhe scheme=bgv",
          heir.Pass("mlir-to-bgv", min_slot_count=8192),
          "scheme-to-openfhe",
      )
      self.assertIn("openfhe.", str(module))

      header = heir.translate(module, "emit-openfhe-pke-header")
      source = heir.translate(
          module, "--emit-openfhe-pke", openfhe_include_type="source-relative"
      )
      self.assertIn("dot_product", header)
      self.assertIn("dot_product", source)
      self.assertIn("src/pke/include/openfhe.h", source)

      # Options must not leak from one translate call into the next.
      default_source = heir.translate(module, "emit-openfhe-pke")
      self.assertNotIn("src/pke/include/openfhe.h", default_source)

  def test_translate_does_not_modify_input(self):
    with heir.context():
      module = heir.parse(DOT_PRODUCT)
      before = str(module)
      heir.translate(module, "emit-function-info")
      self.assertEqual(before, str(module))

  def test_getting_started_example(self):
    # Keep in sync with docs/content/en/docs/getting_started.md.
    with heir.context():
      module = heir.parse(DOT_PRODUCT)
      heir.run(
          module,
          heir.Pass("annotate-module", backend="openfhe", scheme="bgv"),
          heir.Pass("mlir-to-bgv", min_slot_count=8192),
          "scheme-to-openfhe",
      )
      header = heir.translate(module, "emit-openfhe-pke-header")
      source = heir.translate(
          module, "emit-openfhe-pke", openfhe_include_type="install-relative"
      )
      self.assertIn("dot_product", header)
      self.assertIn("openfhe/pke/openfhe.h", source)


class BuildIRTest(unittest.TestCase):

  def test_build_secret_ir(self):
    with heir.module() as m:
      i32 = types.i32()

      @func.FuncOp.from_py_func(i32)
      def conceal_add(x):
        c = arith.ConstantOp(i32, 7)
        y = arith.AddIOp(x, c)
        return secret.ConcealOp(types.secret(i32), y).result

      self.assertTrue(m.operation.verify())
      self.assertIn("secret.conceal %0 : i32 -> !secret.secret<i32>", str(m))

  def test_all_dialects_import(self):
    for name in HEIR_DIALECTS + UPSTREAM_DIALECTS:
      with self.subTest(dialect=name):
        importlib.import_module(f"heir.dialects.{name}")

  def test_heir_ops_are_registered(self):
    with heir.context():
      module = heir.parse("""
        func.func @f(%x: !mod_arith.int<17 : i32>) -> !mod_arith.int<17 : i32> {
          %0 = mod_arith.add %x, %x : !mod_arith.int<17 : i32>
          return %0 : !mod_arith.int<17 : i32>
        }
      """)
      importlib.import_module("heir.dialects.mod_arith")
      add = module.body.operations[0].regions[0].blocks[0].operations[0]
      self.assertEqual(type(add).__name__, "AddOp")


class PassTest(unittest.TestCase):

  def test_pass_str(self):
    self.assertEqual(str(heir.Pass("canonicalize")), "canonicalize")
    self.assertEqual(
        str(heir.Pass("mlir-to-bgv", min_slot_count=8, foo_bar=True)),
        "mlir-to-bgv{min-slot-count=8 foo-bar=true}",
    )
    self.assertEqual(
        str(heir.Pass("p", values=[1, 2], name="a b")),
        'p{values=1,2 name="a b"}',
    )

  def test_pipeline_string(self):
    self.assertEqual(
        heir.pipeline_string(
            "--mlir-to-bgv=min-slot-count=8",
            "--mlir-to-bgv{min-slot-count=8}",
            '--annotate-module="backend=openfhe scheme=bgv"',
            "cse",
        ),
        "builtin.module(mlir-to-bgv{min-slot-count=8},"
        "mlir-to-bgv{min-slot-count=8},"
        "annotate-module{backend=openfhe scheme=bgv},cse)",
    )
    self.assertEqual(
        heir.pipeline_string("builtin.module(cse)"), "builtin.module(cse)"
    )

  def test_run_secret_pass(self):
    with heir.context():
      module = heir.parse("""
        func.func @f(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
          %0 = secret.generic(%arg0: !secret.secret<i32>) {
          ^body(%x: i32):
            %1 = arith.addi %x, %x : i32
            secret.yield %1 : i32
          } -> !secret.secret<i32>
          return %0 : !secret.secret<i32>
        }
      """)
      heir.run(module, "secret-forget-secrets", "canonicalize")
      self.assertNotIn("secret", str(module))

  def test_unknown_pass(self):
    with heir.context():
      module = heir.parse("func.func @f() { return }")
      with self.assertRaises(ValueError):
        heir.run(module, "not-a-real-pass")

  def test_yosys_pipelines_are_not_available(self):
    # The python package ships neither Yosys's techmap files nor ABC.
    with heir.context():
      module = heir.parse("func.func @f() { return }")
      for name in ["mlir-to-cggi", "yosys-optimizer"]:
        with self.subTest(name=name):
          with self.assertRaises(ValueError):
            heir.run(module, name)


class TranslateTest(unittest.TestCase):

  def test_translations(self):
    names = heir.translations()
    for name in ["emit-openfhe-pke", "emit-lattigo", "emit-tfhe-rust"]:
      self.assertIn(name, names)

  def test_unknown_translation(self):
    with heir.context():
      module = heir.parse("func.func @f() { return }")
      with self.assertRaises(heir.TranslationError):
        heir.translate(module, "emit-nothing")

  def test_unknown_option(self):
    with heir.context():
      module = heir.parse("func.func @f() { return }")
      with self.assertRaisesRegex(heir.TranslationError, "unknown option"):
        heir.translate(module, "emit-openfhe-pke", not_an_option=3)
      with self.assertRaisesRegex(heir.TranslationError, "unknown option"):
        heir.translate(module, "emit-openfhe-pke", "not-a-flag")
      # These would print to stdout and exit the process.
      for flags, options in [
          (["--help"], {}),
          (["-h"], {}),
          ([], {"h": True}),
          (["--version"], {}),
          (["--print-options"], {}),
      ]:
        with self.subTest(flags=flags, options=options):
          with self.assertRaisesRegex(
              heir.TranslationError, "unsupported option"
          ):
            heir.translate(module, "emit-openfhe-pke", *flags, **options)

  def test_invalid_option_value(self):
    with heir.context():
      module = heir.parse("func.func @f() { return }")
      # LLVM prints errors about option values to stderr, so the exception
      # should at least name the offending options.
      with self.assertRaisesRegex(
          heir.TranslationError, "--openfhe-include-type=bogus"
      ):
        heir.translate(module, "emit-openfhe-pke", openfhe_include_type="bogus")
      with self.assertRaisesRegex(
          heir.TranslationError, "--openfhe-include-type"
      ):
        heir.translate(module, "emit-openfhe-pke", "--openfhe-include-type")
      # The failed calls must not affect later ones.
      self.assertIn("f", heir.translate(module, "emit-function-info"))

  def test_unverified_ir_raises_translation_error(self):
    with heir.module() as m:
      i32 = types.i32()
      i64 = types.i64()

      @func.FuncOp.from_py_func(i32)
      def bad(x):
        # Mismatched operand types: fails MLIR verification.
        return arith.AddIOp(x, arith.ConstantOp(i64, 1)).result

      with self.assertRaises(heir.TranslationError) as cm:
        heir.translate(m, "emit-openfhe-pke")
      self.assertIn("arith.addi", str(cm.exception))

  def test_translation_failure_has_diagnostics(self):
    with heir.context():
      # emit-openfhe-pke does not support the secret dialect.
      module = heir.parse("""
        func.func @f(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
          return %arg0 : !secret.secret<i32>
        }
      """)
      with self.assertRaises(heir.TranslationError) as cm:
        heir.translate(module, "emit-openfhe-pke")
      self.assertTrue(str(cm.exception))


class TypesTest(unittest.TestCase):

  def test_types(self):
    with heir.context():
      i16 = types.i16()
      self.assertEqual(str(types.tensor([4, 8], i16)), "tensor<4x8xi16>")
      s = types.secret(types.tensor([1024], i16))
      self.assertEqual(str(s), "!secret.secret<tensor<1024xi16>>")
      self.assertTrue(types.is_secret(s))
      self.assertFalse(types.is_secret(i16))
      self.assertEqual(str(types.secret_value_type(s)), "tensor<1024xi16>")
      self.assertEqual(str(types.mod_arith(17)), "!mod_arith.int<17 : i8>")
      self.assertEqual(
          str(types.mod_arith(65537)), "!mod_arith.int<65537 : i32>"
      )
      self.assertEqual(
          str(types.mod_arith(17, types.i64())), "!mod_arith.int<17 : i64>"
      )
      self.assertEqual(str(types.f32()), "f32")
      self.assertEqual(str(types.index()), "index")


if __name__ == "__main__":
  unittest.main()
