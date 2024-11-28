#ifndef LIB_DIALECT_MGMT_IR_MGMTOPS_TD_
#define LIB_DIALECT_MGMT_IR_MGMTOPS_TD_

include "lib/Dialect/Mgmt/IR/MgmtDialect.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Mgmt_Op<string mnemonic, list<Trait> traits = [Pure, ElementwiseMappable, SameOperandsAndResultType]> :
        Op<Mgmt_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::mgmt";
}

def Mgmt_ModReduceOp : Mgmt_Op<"modreduce"> {
  let summary = "Modulus switch the input ciphertext down by one limb (RNS assumed)";

  let description = [{
    This is scheme-agonistic operation that implies modulus switching/rescaling
    by one limb.

    Input ciphertext is assumed to be in RNS form when further lowered.

    When further lowered, it could be lowered to bgv.modulus_switch
    or ckks.rescale depending on the scheme.
  }];

  let arguments = (ins
    AnyType:$input
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def Mgmt_RelinearizeOp : Mgmt_Op<"relinearize"> {
  let summary = "Relinearize the input ciphertext to be _linear_";

  let description = [{
    This is scheme-agonistic operation that implies relinearization
    of the input ciphertext to be _linear_ (i.e. returns to dimension 2).

    This is used solely by multiplication. For rotation, currently HEIR
    assumes relinearization is done internally and does not have a separate
    scheme-specific operation for it.

    This accepts a ciphertext with dimension > 2 and returns a ciphertext
    with dimension 2. Note that the semantic includes the relinearization
    of higher dimension input like input with dimension 4 or higher,
    which when materialized should require multiple relinearization keys.

    When further lowered, it could be lowered to bgv.relinearize
    or ckks.relinearize depending on the scheme.
  }];

  let arguments = (ins
    AnyType:$input
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

#endif  // LIB_DIALECT_MGMT_IR_MGMTOPS_TD_
