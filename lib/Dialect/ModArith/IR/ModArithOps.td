#ifndef LIB_DIALECT_MODARITH_IR_MODARITHOPS_TD_
#define LIB_DIALECT_MODARITH_IR_MODARITHOPS_TD_

include "lib/Dialect/ModArith/IR/ModArithDialect.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class ModArith_Op<string mnemonic, list<Trait> traits = [Pure]> :
        Op<ModArith_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::mod_arith";
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class ModArith_BinaryOp<string mnemonic, list<Trait> traits = []> :
    ModArith_Op<mnemonic, traits # [SameOperandsAndResultType, Pure, ElementwiseMappable]>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs, APIntAttr:$modulus)>,
    Results<(outs SignlessIntegerLike:$output)> {
  let hasVerifier = 1;
  let assemblyFormat ="operands attr-dict `:` type($output)";
}

def ModArith_AddOp : ModArith_BinaryOp<"add", [Commutative]> {
  let summary = "modular addition operation";
  let description = [{
    Computes addition modulo a statically known modulus $q$.
  }];
}

def ModArith_SubOp : ModArith_BinaryOp<"sub"> {
  let summary = "modular subtraction operation";
  let description = [{
    Computes subtraction modulo a statically known modulus $q$.
  }];
}

def ModArith_MulOp : ModArith_BinaryOp<"mul", [Commutative]> {
  let summary = "modular multiplication operation";
  let description = [{
    Computes multiplication modulo a statically known modulus $q$.
  }];
}

def ModArith_MacOp : ModArith_Op<"mac", [SameOperandsAndResultType, Pure, ElementwiseMappable]> {
  let summary = "modular multiplication-and-accumulation operation";

  let description = [{
    `mod_arith.mac x, y, z {modulus = q}` computes $(x * y) + z \mod q$
  }];
  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs, SignlessIntegerLike:$acc, APIntAttr:$modulus);
  let results = (outs SignlessIntegerLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def ModArith_BarrettReduceOp : ModArith_Op<"barrett_reduce", [SameOperandsAndResultType]> {
  let summary = "Compute the first step of the Barrett reduction.";
  let description = [{
    Let $q$ denote a statically known modulus and $b = 4^{w}$, where $w$ is the
    smallest bit-width that contains the range $[0, q)$. The Barrett reduce
    operation computes `barret_reduce x = x - floor(x * floor(b / q) / b) * q`.

    Given $0 <= x < q^2$, then this will compute $(x \mod q)$ or $(x \mod q) + p$.
  }];

  let arguments = (ins
    SignlessIntegerLike:$input,
    APIntAttr:$modulus
  );
  let results = (outs SignlessIntegerLike:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($input))";

  let hasVerifier = 1;
}

def ModArith_SubIfGEOp : ModArith_Op<"subifge", [SameOperandsAndResultType]> {
  let summary = "Compute (x >= y) ? x - y : x.";

  let arguments = (ins
    SignlessIntegerLike:$lhs,
    SignlessIntegerLike:$rhs
  );
  let results = (outs SignlessIntegerLike:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))";
}

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHOPS_TD_
