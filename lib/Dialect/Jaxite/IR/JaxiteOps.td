#ifndef LIB_DIALECT_JAXITE_IR_JAXITEOPS_TD_
#define LIB_DIALECT_JAXITE_IR_JAXITEOPS_TD_

include "JaxiteDialect.td"
include "JaxiteTypes.td"

include "lib/Dialect/LWE/IR/LWETypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Jaxite_Op<string mnemonic, list<Trait> traits = []> :
        Op<Jaxite_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::heir::jaxite";
}

def ConstantOp : Jaxite_Op<"constant", [Pure]> {
  let arguments = (ins I1:$value, Jaxite_Params:$params);
  let results = (outs LWECiphertext:$output);
}

def Lut3Op : Jaxite_Op<"lut3", [
    Pure,
    AllTypesMatch<["a", "b", "c", "output"]>
]> {
  let description = [{
    The operation computed by this function can be
    interpreted as

      truth_table >> {c, b, a}

    where {c, b, a} is the unsigned 3-bit integer with bits c, b, a from most
    significant bit to least-significant bit.
  }];
  let arguments = (
    ins LWECiphertext:$a, LWECiphertext:$b, LWECiphertext:$c,
    I8:$truth_table,
    Jaxite_ServerKeySet:$serverKeySet,
    Jaxite_Params:$params
  );
  let results = (outs LWECiphertext:$output);
}

#endif  // LIB_DIALECT_JAXITE_IR_JAXITEOPS_TD_
