#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_
#define LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_

include "OpenfheDialect.td"
include "OpenfheTypes.td"

include "lib/Utils/Tablegen/InPlaceOpInterface.td"
include "lib/Dialect/HEIRInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Openfhe_Op<string mnemonic, list<Trait> traits = []> :
        Op<Openfhe_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::heir::openfhe";
}

class Openfhe_UnaryOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_Op<mnemonic, traits # [Pure, AllTypesMatch<["ciphertext", "output"]>]>{
 let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext
  );
  let results = (outs Openfhe_Ciphertext:$output);
}

// TODO(#2532): consider removing the op result
class Openfhe_UnaryInPlaceOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_Op<mnemonic, traits # [InPlaceOpInterface]>{
 let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext
  );
  // The "result" of an in-place op is a shallow reference to the mutated
  // operand.
  let results = (outs Openfhe_Ciphertext:$output);
  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 1; }";
}

class Openfhe_BinaryOp<string mnemonic, list<Trait> traits = []>
  :  Openfhe_Op<mnemonic, traits # [
    Pure, AllTypesMatch<["lhs", "rhs", "output"]>
]>{
 let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$lhs,
    Openfhe_Ciphertext:$rhs
  );
  let results = (outs Openfhe_Ciphertext:$output);
}

def Openfhe_PlaintextOrCiphertext : AnyTypeOf<[Openfhe_Ciphertext, Openfhe_Plaintext]>;

// Ciphertext-plaintext binary op
class Openfhe_BinaryPlainOp<string mnemonic, list<Trait> traits = []>
  :  Openfhe_Op<mnemonic, traits # [Pure]>{
 let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_PlaintextOrCiphertext:$lhs,
    Openfhe_PlaintextOrCiphertext:$rhs
  );
  let results = (outs Openfhe_Ciphertext:$output);

  let extraClassDeclaration = [{
    int getCiphertextOperandIndex() {
      if (isa<::mlir::heir::openfhe::CiphertextType>(getLhs().getType())) {
        return 1;
      }
      if (isa<::mlir::heir::openfhe::CiphertextType>(getRhs().getType())) {
        return 2;
      }
      assert(false && "ciphertext-plaintext op has no ciphertext operand");
      return -1;
    }
    int getPlaintextOperandIndex() {
      if (isa<::mlir::heir::openfhe::PlaintextType>(getLhs().getType())) {
        return 1;
      }
      if (isa<::mlir::heir::openfhe::PlaintextType>(getRhs().getType())) {
        return 2;
      }
      assert(false && "ciphertext-plaintext op has no plaintext operand");
      return -1;
    }

    ::mlir::Value getCiphertext() {
      return getOperands()[getCiphertextOperandIndex()];
    }

    ::mlir::Value getPlaintext() {
      return getOperands()[getPlaintextOperandIndex()];
    }
  }];
}

class Openfhe_BinaryInPlaceOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_Op<mnemonic, traits # [InPlaceOpInterface, AllTypesMatch<["lhs", "rhs"]>]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$lhs,
    Openfhe_Ciphertext:$rhs
  );
  // The "result" of an in-place op is a shallow reference to the mutated
  // operand.
  let results = (outs Openfhe_Ciphertext:$output);

  // Always mutate the LHS
  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 1; }";
}

// Ciphertext-plaintext in-place binary op
class Openfhe_BinaryPlainInPlaceOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_Op<mnemonic, traits # [InPlaceOpInterface]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_PlaintextOrCiphertext:$lhs,
    Openfhe_PlaintextOrCiphertext:$rhs
  );
  // The "result" of an in-place op is a shallow reference to the mutated
  // operand.
  let results = (outs Openfhe_Ciphertext:$output);

  let extraClassDeclaration = [{
    int getCiphertextOperandIndex() {
      if (isa<::mlir::heir::openfhe::CiphertextType>(getLhs().getType())) {
        return 1;
      }
      if (isa<::mlir::heir::openfhe::CiphertextType>(getRhs().getType())) {
        return 2;
      }
      assert(false && "ciphertext-plaintext op has no ciphertext operand");
      return -1;
    }
    int getPlaintextOperandIndex() {
      if (isa<::mlir::heir::openfhe::PlaintextType>(getLhs().getType())) {
        return 1;
      }
      if (isa<::mlir::heir::openfhe::PlaintextType>(getRhs().getType())) {
        return 2;
      }
      assert(false && "ciphertext-plaintext op has no plaintext operand");
      return -1;
    }

    ::mlir::Value getCiphertext() {
      return getOperands()[getCiphertextOperandIndex()];
    }

    ::mlir::Value getPlaintext() {
      return getOperands()[getPlaintextOperandIndex()];
    }

    // Always mutate the Ciphertext operand
    int getInPlaceOperandIndex() {
      return getCiphertextOperandIndex();
    }
  }];
}

def GenParamsOp : Openfhe_Op<"gen_params"> {
  let description = [{
    Generates the parameters for the OpenFHE scheme.

    `mulDepth` is the depth of the multiplication circuit,
    including the bootstrapping depth.

    `plainMod` is the modulus of the plaintext space. If we
    are using CKKS, this is 0.

    `insecure` is a flag that determines whether the parameters
    are generated securely or not. In Openfhe, this means setting
    HEStd_NotSet for security level.

    For other flags, see the OpenFHE documentation in
    https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/README.md#description-of-the-cryptocontext-parameters-and-their-restrictions
  }];
  let arguments = (ins
    // Essential parameters
    I64Attr:$mulDepth,
    I64Attr:$plainMod,
    // Optional parameters
    DefaultValuedAttr<I64Attr, "0">:$ringDim,
    DefaultValuedAttr<I64Attr, "0">:$batchSize,
    // Modulus chain parameters
    DefaultValuedAttr<I64Attr, "0">:$firstModSize,
    DefaultValuedAttr<I64Attr, "0">:$scalingModSize,
    // KPZ21 way of noise estimation
    DefaultValuedAttr<I64Attr, "0">:$evalAddCount,
    DefaultValuedAttr<I64Attr, "0">:$keySwitchCount,
    // Key switching technique parameters
    DefaultValuedAttr<I64Attr, "0">:$digitSize,
    DefaultValuedAttr<I64Attr, "0">:$numLargeDigits,
    // Relinearization parameters
    DefaultValuedAttr<I64Attr, "0">:$maxRelinSkDeg,
    // Option switches
    DefaultValuedAttr<BoolAttr, "false">:$insecure,
    DefaultValuedAttr<BoolAttr, "false">:$encryptionTechniqueExtended,
    DefaultValuedAttr<BoolAttr, "false">:$keySwitchingTechniqueBV,
    DefaultValuedAttr<BoolAttr, "false">:$scalingTechniqueFixedManual
  );
  let results = (outs Openfhe_CCParams:$params);
}

def GenContextOp : Openfhe_Op<"gen_context"> {
  let arguments = (ins
    Openfhe_CCParams:$params,
    BoolAttr:$supportFHE
  );
  let results = (outs Openfhe_CryptoContext:$context);
}

def GenMulKeyOp : Openfhe_Op<"gen_mulkey"> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_PrivateKey:$privateKey
  );
}

def GenRotKeyOp : Openfhe_Op<"gen_rotkey"> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_PrivateKey:$privateKey,
    DenseI64ArrayAttr:$indices
  );
}

def SetupBootstrapOp : Openfhe_Op<"setup_bootstrap"> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Builtin_IntegerAttr:$levelBudgetEncode,
    Builtin_IntegerAttr:$levelBudgetDecode
  );
}

def GenBootstrapKeyOp : Openfhe_Op<"gen_bootstrapkey"> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_PrivateKey:$privateKey
  );
}

def MakePackedPlaintextOp : Openfhe_Op<"make_packed_plaintext", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RankedTensorOf<[AnyInteger]>:$value
  );
  let results = (outs Openfhe_Plaintext:$plaintext);
}

def MakeCKKSPackedPlaintextOp : Openfhe_Op<"make_ckks_packed_plaintext", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RankedTensorOf<[AnyFloat, AnyInteger]>:$value
  );
  let results = (outs Openfhe_Plaintext:$plaintext);
}

def EncryptOp : Openfhe_Op<"encrypt", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Plaintext:$plaintext,
    Openfhe_PublicKeyOrPrivateKey:$encryptionKey
  );
  let results = (outs Openfhe_Ciphertext:$ciphertext);
}

def DecryptOp : Openfhe_Op<"decrypt", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext,
    Openfhe_PrivateKey:$privateKey
  );
  let results = (outs Openfhe_Plaintext:$plaintext);
}

def AddOp : Openfhe_BinaryOp<"add"> {
  let summary = "Addition of two ciphertexts.";
}

def AddInPlaceOp : Openfhe_BinaryInPlaceOp<"add_inplace"> {
  let summary = "In-place addition, modifying lhs.";
}

def SubOp : Openfhe_BinaryOp<"sub"> {
  let summary = "Subtraction of two ciphertexts.";
}

def SubInPlaceOp : Openfhe_BinaryInPlaceOp<"sub_inplace"> {
  let summary = "In-place subtraction, modifying lhs.";
}

def AddPlainOp : Openfhe_BinaryPlainOp<"add_plain"> {
  let summary = "Addition of a ciphertext and a plaintext.";
}

def AddPlainInPlaceOp : Openfhe_BinaryPlainInPlaceOp<"add_plain_inplace"> {
  let summary = "In-place addition of a ciphertext and a plaintext, modifying the ciphertext operand.";
}

def SubPlainOp : Openfhe_BinaryPlainOp<"sub_plain"> {
  let summary = "Subtraction of a ciphertext and a plaintext.";
}

def SubPlainInPlaceOp : Openfhe_BinaryPlainInPlaceOp<"sub_plain_inplace"> {
  let summary = "In-place subtraction of a ciphertext and a plaintext, modifying the ciphertext operand.";
}

def MulOp : Openfhe_BinaryOp<"mul", [IncreasesMulDepthOpInterface]> {
  let summary = "Multiplication of two ciphertexts with relinearization.";
}

// There appears to be no EvalMultInPlace for ct-ct or ct-pt (just ct-const),
// with or without relinearization.
def MulNoRelinOp : Openfhe_BinaryOp<"mul_no_relin", [IncreasesMulDepthOpInterface]> {
  let summary = "Multiplication of two ciphertexts without relinearization.";
}

def MulPlainOp : Openfhe_BinaryPlainOp<"mul_plain", [Pure, IncreasesMulDepthOpInterface]> {
  let summary = "Multiplication of a ciphertext and a plaintext.";
}

def MulConstOp : Openfhe_Op<"mul_const",[
    Pure,
    AllTypesMatch<["ciphertext", "output"]>
]> {
  let summary = "Multiplication of a ciphertext and a constant.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext,
    I64:$constant
  );
  let results = (outs Openfhe_Ciphertext:$output);
}

def MulConstInPlaceOp : Openfhe_Op<"mul_const_inplace", [InPlaceOpInterface]> {
  let summary = "In-place multiplication of a ciphertext and a constant, modifying the ciphertext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext,
    I64:$constant
  );
  // The "result" of an in-place op is a shallow reference to the mutated
  // operand.
  let results = (outs Openfhe_Ciphertext:$output);
  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 1; }";
}

def NegateOp : Openfhe_UnaryOp<"negate"> { let summary = "Negation of a ciphertext."; }
def NegateInPlaceOp : Openfhe_UnaryInPlaceOp<"negate_inplace"> { let summary = "In-place negation of a ciphertext."; }
def SquareOp : Openfhe_UnaryOp<"square"> { let summary = "Square of a ciphertext."; }
def SquareInPlaceOp : Openfhe_UnaryInPlaceOp<"square_inplace"> { let summary = "In-place square of a ciphertext."; }
def RelinOp : Openfhe_UnaryOp<"relin"> { let summary = "Relinearization of a ciphertext."; }
def RelinInPlaceOp : Openfhe_UnaryInPlaceOp<"relin_inplace"> { let summary = "In-place relinearization of a ciphertext."; }
def ModReduceOp : Openfhe_UnaryOp<"mod_reduce"> { let summary = "mod_reduce of a ciphertext. (used only for BGV/CKKS)"; }
def ModReduceInPlaceOp : Openfhe_UnaryInPlaceOp<"mod_reduce_inplace"> { let summary = "In-place mod_reduce of a ciphertext. (used only for BGV/CKKS)"; }

def LevelReduceOp : Openfhe_UnaryOp<"level_reduce"> {
  let summary = "Reduce the level of a ciphertext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext,
    DefaultValuedAttr<I64Attr, "1">:$levelToDrop
  );
}

def LevelReduceInPlaceOp : Openfhe_UnaryInPlaceOp<"level_reduce_inplace",
    [InPlaceOpInterface]> {
  let summary = "In-place level reduction of a ciphertext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext,
    DefaultValuedAttr<I64Attr, "1">:$levelToDrop
  );
  // The "result" of an in-place op is a shallow reference to the mutated
  // operand.
  let results = (outs Openfhe_Ciphertext:$output);
  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 1; }";
}

// no in-place variant for EvalRotate
def RotOp : Openfhe_Op<"rot", [
  Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext,
    Builtin_IntegerAttr:$index
  );
  let results = (outs Openfhe_Ciphertext:$output);
}

// no in-place variant for EvalAutomorphism
def AutomorphOp : Openfhe_Op<"automorph", [
  Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey
  );
  let results = (outs Openfhe_Ciphertext:$output);
}

def KeySwitchOp : Openfhe_Op<"key_switch", [
   Pure,
   AllTypesMatch<["ciphertext", "output"]>
]> {
  let summary = "Key switch a ciphertext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey
  );
  let results = (outs Openfhe_Ciphertext:$output);
}

def KeySwitchInPlaceOp : Openfhe_Op<"key_switch_inplace",
    [InPlaceOpInterface]> {
  let summary = "In-place key switch of a ciphertext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey
  );
  // The "result" of an in-place op is a shallow reference to the mutated
  // operand.
  let results = (outs Openfhe_Ciphertext:$output);
  let extraClassDeclaration = "int getInPlaceOperandIndex() { return 1; }";
}

def KeySwitchDownOp : Openfhe_Op<"key_switch_down", [Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let summary = "Key switch down from extended (P*Q) basis to normal (Q) basis.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$ciphertext
  );
  let results = (outs Openfhe_Ciphertext:$output);
}

// No in-place bootstrap variant
def BootstrapOp : Openfhe_UnaryOp<"bootstrap", [Pure, ResetsMulDepthOpInterface]> {
  let summary = "OpenFHE bootstrap operation of a ciphertext. (For CKKS)";
}

// No in-place fast rotation variants
def FastRotationPrecomputeOp : Openfhe_Op<"fast_rotation_precompute", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$input
  );
  let results = (outs Openfhe_DigitDecomposition:$output);
}

// TODO(#2519): support the "Ext" variant for hybrid key switching.
def FastRotationOp : Openfhe_Op<"fast_rotation", [Pure,
  AllTypesMatch<["input", "output"]>,
  BatchVectorizableOpInterface,
  // The operation also supports batching many rotations of the same input.
  DeclareOpInterfaceMethods<ElementwiseByOperandOpInterface>,
  DeclareOpInterfaceMethods<
      BatchVectorizableOpInterface,
      ["isBatchCompatible", "buildBatchedOperation"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$input,
    Index:$index,
    IndexAttr:$cyclotomicOrder,
    Openfhe_DigitDecomposition:$precomputedDigitDecomp
  );
  let results = (outs Openfhe_Ciphertext:$output);
}

def FastRotationExtOp : Openfhe_Op<"fast_rotation_ext", [Pure,
  AllTypesMatch<["input", "output"]>,
  BatchVectorizableOpInterface,
  DeclareOpInterfaceMethods<ElementwiseByOperandOpInterface>,
  DeclareOpInterfaceMethods<
      BatchVectorizableOpInterface,
      ["isBatchCompatible", "buildBatchedOperation"]>
]> {
  let summary = "Fast (hoisted) rotation in extended (P*Q) CRT basis.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_Ciphertext:$input,
    Index:$index,
    Openfhe_DigitDecomposition:$precomputedDigitDecomp,
    DefaultValuedAttr<BoolAttr, "true">:$addFirst
  );
  let results = (outs Openfhe_Ciphertext:$output);
}

def DecodeOp : Openfhe_Op<"decode", [
  Pure,
]> {
  let arguments = (ins Openfhe_Plaintext:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    operands attr-dict `:` type($input) `->` type($output)
  }];
}

def DecodeCKKSOp : Openfhe_Op<"decode_ckks", [
  Pure,
]> {
  let arguments = (ins Openfhe_Plaintext:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    operands attr-dict `:` type($input) `->` type($output)
  }];
}

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_
