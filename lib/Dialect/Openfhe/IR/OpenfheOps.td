#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_
#define LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_

include "OpenfheDialect.td"
include "OpenfheTypes.td"

include "lib/Dialect/LWE/IR/LWETypes.td"
include "lib/Dialect/LWE/IR/LWETraits.td"
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

class Openfhe_UnaryTypeSwitchOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_Op<mnemonic, traits # [
    Pure,
]>{
 let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext
  );
  let results = (outs LWECiphertext:$output);
}

class Openfhe_UnaryOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_UnaryTypeSwitchOp<
      mnemonic, traits # [AllTypesMatch<["ciphertext", "output"]>]
    >;

class Openfhe_BinaryOp<string mnemonic, list<Trait> traits = []>
  :  Openfhe_Op<mnemonic, traits # [
    Pure
]>{
 let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$lhs,
    LWECiphertext:$rhs
  );
  let results = (outs LWECiphertext:$output);
}

class Openfhe_BinaryInPlaceOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_Op<mnemonic, traits # [
    AllTypesMatch<["lhs", "rhs"]>,
    ]> {

  let summary = "In-place binary operation for OpenFHE";

  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$lhs,
    LWECiphertext:$rhs
  );
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
    RankedTensorOf<[AnyInteger]>:$value)
  ;
  let results = (outs LWEPlaintext:$plaintext);
  let hasVerifier = 1;
}

def MakeCKKSPackedPlaintextOp : Openfhe_Op<"make_ckks_packed_plaintext", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RankedTensorOf<[AnyFloat, AnyInteger]>:$value)
  ;
  let results = (outs LWEPlaintext:$plaintext);
  let hasVerifier = 1;
}

def EncryptOp : Openfhe_Op<"encrypt", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWEPlaintext:$plaintext,
    Openfhe_PublicKeyOrPrivateKey:$encryptionKey)
  ;
  let results = (outs LWECiphertext:$ciphertext);
}

def DecryptOp : Openfhe_Op<"decrypt", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext,
    Openfhe_PrivateKey:$privateKey)
  ;
  let results = (outs LWEPlaintext:$plaintext);
}

def AddOp : Openfhe_BinaryOp<"add",
    [SameOperandsAndResultRings,
    InferTypeOpAdaptor]> {
  let summary = "OpenFHE add operation of two ciphertexts.";
}
def SubOp : Openfhe_BinaryOp<"sub",
    [SameOperandsAndResultRings,
    InferTypeOpAdaptor]> {
  let summary = "OpenFHE sub operation of two ciphertexts.";
}

// In-Place Addition
def AddInPlaceOp : Openfhe_BinaryInPlaceOp<"add_inplace"> {
  let summary = "Performs in-place homomorphic addition,  modifying lhs.";
}

// In-Place Subtraction
def SubInPlaceOp : Openfhe_BinaryInPlaceOp<"sub_inplace"> {
  let summary = "Performs in-place homomorphic subtraction, modifying lhs.";
}


def AddPlainOp : Openfhe_Op<"add_plain",[
    Pure,
    AllCiphertextTypesMatch,
    InferTypeOpAdaptor
]> {
  let summary = "OpenFHE add operation of a ciphertext and a plaintext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWEPlaintextOrCiphertext:$lhs,
    LWEPlaintextOrCiphertext:$rhs
  );
  let results = (outs LWECiphertext:$output);
}

def SubPlainOp : Openfhe_Op<"sub_plain",[
    Pure,
    AllCiphertextTypesMatch,
    InferTypeOpAdaptor
]> {
  let summary = "OpenFHE sub operation of a ciphertext and a plaintext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWEPlaintextOrCiphertext:$lhs,
    LWEPlaintextOrCiphertext:$rhs
  );
  let results = (outs LWECiphertext:$output);
}

def MulOp : Openfhe_BinaryOp<"mul"> { let summary = "OpenFHE mul operation of two ciphertexts with relinearization."; }

def MulNoRelinOp : Openfhe_Op<"mul_no_relin", [Pure, SameOperandsAndResultRings, InferTypeOpAdaptor]> {
  let summary = "OpenFHE mul operation of two ciphertexts without relinearization.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$lhs,
    LWECiphertext:$rhs
  );
  let results = (outs LWECiphertext:$output);
  let hasVerifier = 1;
}

def MulPlainOp : Openfhe_Op<"mul_plain",[
    Pure
]> {
  let summary = "OpenFHE mul operation of a ciphertext and a plaintext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext,
    LWEPlaintext:$plaintext
  );
  let results = (outs LWECiphertext:$output);
}

def MulConstOp : Openfhe_Op<"mul_const",[
    Pure,
    AllTypesMatch<["ciphertext", "output"]>
]> {
  let summary = "OpenFHE mul operation of a ciphertext and a constant.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext,
    I64:$constant
  );
  let results = (outs LWECiphertext:$output);
}

def NegateOp : Openfhe_UnaryOp<"negate"> { let summary = "OpenFHE negate operation of a ciphertext."; }
def SquareOp : Openfhe_UnaryOp<"square"> { let summary = "OpenFHE square operation of a ciphertext."; }
def RelinOp : Openfhe_UnaryTypeSwitchOp<"relin"> { let summary = "OpenFHE relinearize operation of a ciphertext."; }

def ModReduceOp : Openfhe_UnaryTypeSwitchOp<"mod_reduce"> { let summary = "OpenFHE mod_reduce operation of a ciphertext. (used only for BGV/CKKS)"; }
def LevelReduceOp : Openfhe_UnaryTypeSwitchOp<"level_reduce"> {
  let summary = "OpenFHE level_reduce operation of a ciphertext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext,
    DefaultValuedAttr<I64Attr, "1">:$levelToDrop
  );
}

def RotOp : Openfhe_Op<"rot", [
  Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext,
    Builtin_IntegerAttr:$index
  );
  let results = (outs LWECiphertext:$output);
}

def AutomorphOp : Openfhe_Op<"automorph", [
  Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey
  );
  let results = (outs LWECiphertext:$output);
}

def KeySwitchOp : Openfhe_Op<"key_switch", [
   Pure,
   AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey
  );
  let results = (outs LWECiphertext:$output);
}

def BootstrapOp : Openfhe_UnaryTypeSwitchOp<"bootstrap"> { let summary = "OpenFHE bootstrap operation of a ciphertext. (For CKKS)"; }

def FastRotationPrecomputeOp : Openfhe_Op<"fast_rotation_precompute", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$input
  );
  let results = (outs Openfhe_DigitDecomposition:$output);
}

// TODO(#1924): support the "Ext" variant for hybrid key switching.
def FastRotationOp : Openfhe_Op<"fast_rotation", [Pure,
   AllTypesMatch<["input", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    LWECiphertext:$input,
    IndexAttr:$index,
    IndexAttr:$cyclotomicOrder,
    Openfhe_DigitDecomposition:$precomputedDigitDecomp
  );
  let results = (outs LWECiphertext:$output);
}

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_
