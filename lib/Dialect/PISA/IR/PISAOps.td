#ifndef LIB_DIALECT_PISA_IR_PISAOPS_TD_
#define LIB_DIALECT_PISA_IR_PISAOPS_TD_

include "lib/Dialect/PISA/IR/PISADialect.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// We only accept tensors of mod_arith with 32-bit typed moduli.
// Note that we do NOT allow moduli that are concretely less than 32 bits but have a larger type (e.g., I64)
// as those allow the compiler to emit code that relies on temporarily using up to 64 bits before mod-reducing.
def Tensor8192I32 : TypeConstraint<CPred<[{
  mlir::isa<mlir::RankedTensorType>($_self) &&
  mlir::cast<mlir::RankedTensorType>($_self).getRank() == 1 &&
  mlir::cast<mlir::RankedTensorType>($_self).getDimSize(0) == 8192 &&
  llvm::isa<mlir::heir::mod_arith::ModArithType>(mlir::cast<mlir::RankedTensorType>($_self).getElementType()) &&
  mlir::cast<mlir::heir::mod_arith::ModArithType>(mlir::cast<mlir::RankedTensorType>($_self).getElementType()).getModulus().getType().isInteger(32)
}]>, "tensor<8192xmod_arith.int< ... : i32>>">;

class PISA_Op<string mnemonic, list<Trait> traits = [Pure]> :
        Op<PISA_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::pisa";
}

class PISA_BinaryOp<string mnemonic, list<Trait> traits = []> :
        PISA_Op<mnemonic, traits # [SameOperandsAndResultType]>,
    Arguments<(ins Tensor8192I32:$lhs, Tensor8192I32:$rhs, I32Attr:$q, I32Attr:$i)>,
    Results<(outs Tensor8192I32:$output)> {
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
  }

def PISA_AddOp : PISA_BinaryOp<"add", [Commutative]> {
  let summary = "addition operation";
  let description = [{
    Computes addition of two polynomials (irrespective of ntt/coefficient representation).
  }];
}

def PISA_SubOp : PISA_BinaryOp<"sub", []> {
  let summary = "subtraction operation";
  let description = [{
    Computes subtraction of two polynomials (irrespective of ntt/coefficient representation).
  }];
}

def PISA_MulOp : PISA_BinaryOp<"mul", [Commutative]> {
  let summary = "multiplication operation";
  let description = [{
    Computes addition of two polynomials (in ntt representation).
  }];
}

def PISA_MuliOp : PISA_Op<"muli", [SameOperandsAndResultType]> {
  let summary = "multiplication-with-immediate operation";
  let description = [{
    Computes multiplication of a polynomial (in ntt representation) with a constant.
  }];
  let arguments = (ins Tensor8192I32:$lhs, I32Attr:$q, I32Attr:$i, I32Attr:$imm);
  let results = (outs Tensor8192I32:$output);
  let assemblyFormat = "$lhs attr-dict `:` qualified(type($output))";
}

def PISA_MacOp : PISA_Op<"mac", [SameOperandsAndResultType]> {
  let summary = "multiply-and-accumulate operation";
  let description = [{
    Computes multiplication of two polynomials (in ntt representation) and adds the result to a third polynomial.
  }];
  let arguments = (ins Tensor8192I32:$lhs, Tensor8192I32:$rhs, Tensor8192I32:$acc, I32Attr:$q, I32Attr:$i);
  let results = (outs Tensor8192I32:$output);
  let assemblyFormat = "$lhs `,` $rhs `,` $acc  attr-dict `:` qualified(type($output))";
}

def PISA_MaciOp : PISA_Op<"maci", [SameOperandsAndResultType]> {
  let summary = "multiply-and-accumulate-with-immediate operation";
  let description = [{
    Computes multiplication of a polynomial (in ntt representation) with a constant and adds the result to a third polynomial.
  }];
  let arguments = (ins Tensor8192I32:$lhs, Tensor8192I32:$acc, I32Attr:$q, I32Attr:$i, I32Attr:$imm);
  let results = (outs Tensor8192I32:$output);
  let assemblyFormat = "$lhs `,` $acc  attr-dict `:` qualified(type($output))";
}

def PISA_NTTOp : PISA_Op<"ntt", [SameOperandsAndResultType]> {
  let summary = "number-theoretic-transform operation";
  let description = [{
    Computes number-theoretic-transform of a polynomial.
  }];
  let arguments = (ins Tensor8192I32:$poly, Tensor8192I32:$w, I32Attr:$q, I32Attr:$i);
  let results = (outs Tensor8192I32:$output);
  let assemblyFormat = "$poly `,` $w attr-dict `:` qualified(type($output))";
}

def PISA_INTTOp : PISA_Op<"intt", [SameOperandsAndResultType]> {
  let summary = "inverse number-theoretic-transform operation";
  let description = [{
    Computes inverse number-theoretic-transform of a polynomial.
  }];
  let arguments = (ins Tensor8192I32:$poly, Tensor8192I32:$w, I32Attr:$q, I32Attr:$i);
  let results = (outs Tensor8192I32:$output);
  let assemblyFormat = "$poly `,` $w attr-dict `:` qualified(type($output))";
}


#endif  // LIB_DIALECT_PISA_IR_PISAOPS_TD_
