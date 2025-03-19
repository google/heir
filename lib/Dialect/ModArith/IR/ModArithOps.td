#ifndef LIB_DIALECT_MODARITH_IR_MODARITHOPS_TD_
#define LIB_DIALECT_MODARITH_IR_MODARITHOPS_TD_

include "lib/Dialect/ModArith/IR/ModArithDialect.td"
include "lib/Dialect/ModArith/IR/ModArithTypes.td"
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

// type conversion operations
def ModArith_EncapsulateOp : ModArith_Op<"encapsulate", [Pure, ElementwiseMappable]> {
  let summary = "encapsulate an integer into a mod_arith type";

  let description = [{
    `mod_arith.encapsulate` converts the integer to be of mod_arith type.

    Examples:
    ```
    mod_arith.encapsulate %c0 : i32 -> mod_arith.int<65537 : i32>
    mod_arith.encapsulate %c1 : i64 -> mod_arith.int<65537>
    ```
  }];

  let arguments = (ins
    SignlessIntegerLike:$input
  );
  let results = (outs ModArithLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

def ModArith_ExtractOp : ModArith_Op<"extract", [Pure, ElementwiseMappable]> {
  let summary = "extract the integer stored inside mod_arith type";

  let description = [{
    `mod_arith.extract` extracts the integer inside the mod_arith type.

    It is required that the bitwidth of the output integer type is the same
    as that of the storage type of the input mod_arith type.

    Examples:
    ```
    %m0 = mod_arith.encapsulate %c0 : i32 -> mod_arith.int<65537 : i32>
    %m1 = mod_arith.encapsulate %c1 : i64 -> mod_arith.int<65537>
    %c2 = mod_arith.extract %m0 : mod_arith.int<65537 : i32> -> i32
    %c3 = mod_arith.extract %m1 : mod_arith.int<65537> -> i64
    ```
  }];

  let arguments = (ins
    ModArithLike:$input
  );
  let results = (outs SignlessIntegerLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

def ModArith_ModSwitchOp : ModArith_Op<"mod_switch", [Pure, SameOperandsAndResultShape]> {
  let summary = "change the modulus of a mod_arith";

  let description = [{
    "mod_switch" operation to change the modulus of a mod_arith type to a
    bigger space.

    Examples:
    ```
    `mod_arith.mod_switch %c0 : mod_arith.int<65537 : i32> to mod_arith.int<65539 : i32>`
    ```
  }];

  let arguments = (ins ModArith_ModArithType:$input);
  let results = (outs ModArith_ModArithType:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

def ModArith_ConstantOp : Op<ModArith_Dialect, "constant",
    [Pure, ConstantLike]> {
  let summary = "Define a constant value via an attribute.";
  let description = [{
    Example:

    ```mlir
    %0 = mod_arith.constant 123 : !mod_arith.int<65537:i32>
    ```
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs ModArithLike:$output);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasFolder = 1;
}


def ModArith_ReduceOp : ModArith_Op<"reduce", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let summary = "reduce the mod arith type to its canonical representative";

  let description = [{
    `mod_arith.reduce x` produces $y$, the canonical representative in $[0, q)$
    such that $x \equiv y \mod q$.

    Examples:
    ```
    %c0 = arith.constant 65538 : i32
    %m0 = mod_arith.encapsulate %c0 : i32 -> mod_arith.int<65537 : i32>
    // mod_arith.extract %m0 produces 65538
    %m1 = mod_arith.reduce %m0 : mod_arith.int<65537: i32>
    // mod_arith.extract %m1 produces 1
    ```
  }];

  let arguments = (ins
    ModArithLike:$input
  );
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

class ModArith_BinaryOp<string mnemonic, list<Trait> traits = []> :
    ModArith_Op<mnemonic, traits # [SameOperandsAndResultType, Pure, ElementwiseMappable]>,
    Arguments<(ins ModArithLike:$lhs, ModArithLike:$rhs)>,
    Results<(outs ModArithLike:$output)> {
  let hasVerifier = 1;
  let assemblyFormat ="operands attr-dict `:` type($output)";
}

def ModArith_AddOp : ModArith_BinaryOp<"add", [Commutative]> {
  let summary = "modular addition operation";
  let description = [{
    Computes modular addition.

    Unless otherwise specified, the operation assumes both inputs are canonical
    representatives and guarantees the output being canonical representative.
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def ModArith_SubOp : ModArith_BinaryOp<"sub"> {
  let summary = "modular subtraction operation";
  let description = [{
    Computes modular subtraction.

    Unless otherwise specified, the operation assumes both inputs are canonical
    representatives and guarantees the output being canonical representative.
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def ModArith_MulOp : ModArith_BinaryOp<"mul", [Commutative]> {
  let summary = "modular multiplication operation";
  let description = [{
    Computes modular multiplication.

    Unless otherwise specified, the operation assumes both inputs are canonical
    representatives and guarantees the output being canonical representative.
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def ModArith_MacOp : ModArith_Op<"mac", [SameOperandsAndResultType, Pure, ElementwiseMappable]> {
  let summary = "modular multiplication-and-accumulation operation";

  let description = [{
    `mod_arith.mac x, y, z` computes $(x * y) + z$

    Unless otherwise specified, the operation assumes all inputs are canonical
    representatives and guarantees the output being canonical representative.
  }];
  let arguments = (ins ModArithLike:$lhs, ModArithLike:$rhs, ModArithLike:$acc);
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// TODO(#1084): migrate barrett/subifge to mod arith type
def ModArith_BarrettReduceOp : ModArith_Op<"barrett_reduce", [SameOperandsAndResultType]> {
  let summary = "Compute the first step of the Barrett reduction.";
  let description = [{
    Let $q$ denote a statically known modulus and $b = 4^{w}$, where $w$ is the
    smallest bit-width that contains the range $[0, q)$. The Barrett reduce
    operation computes `barret_reduce x = x - floor(x * floor(b / q) / b) * q`.

    Given $0 <= x < q^2$, then this will compute $(x \mod q)$ or $(x \mod q) + q$.
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
