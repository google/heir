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

  let hasCanonicalizer = 1;
}

def Mgmt_LevelReduceOp : Mgmt_Op<"level_reduce"> {
  let summary = "Reduce the level of input ciphertext via drop the last k limbs (RNS assumed)";

  let description = [{
    This is scheme-agonistic operation that implies dropping the level
    by k, k specified by the `levelToDrop` attribute.

    If `levelToDrop` is not specified, it defaults to 1.

    Input ciphertext is assumed to be in RNS form when further lowered.

    When further lowered, it could be lowered to bgv.level_reduce
    or ckks.level_reduce depending on the scheme.
  }];

  let arguments = (ins
    AnyType:$input,
    DefaultValuedAttr<I64Attr, "1">:$levelToDrop
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";

  let hasCanonicalizer = 1;
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

def Mgmt_BootstrapOp : Mgmt_Op<"bootstrap"> {
  let summary = "Bootstrap the input ciphertext to refresh its noise budget";

  let description = [{
    This is a scheme-agnostic operation that implies bootstrapping
    of the input ciphertext to refresh its noise budget.

    Bootstrapping is a technique used in homomorphic encryption to
    reduce the noise in a ciphertext, allowing further operations
    to be performed on it without decryption.

    When further lowered, it could be lowered to bgv.bootstrap
    or ckks.bootstrap depending on the scheme.

    For the current backend, only ckks.bootstrap is supported.
    Further backend may include bgv.bootstrap.
  }];

  let arguments = (ins
    AnyType:$input
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def Mgmt_AdjustScaleOp : Mgmt_Op<"adjust_scale"> {
  let summary = "Adjust the scale of the input ciphertext to the specified scale (for BGV and CKKS)";

  let description = [{
    This is scheme-agonistic operation that adjust the scale of the input
    ciphertext to the specified scale.

    The scale is specified by the `scale` attribute.

    At the time of secret-insert-mgmt-{scheme}, the concrete scale
    is not known as the scheme parameter is not generated. After that pass
    the scale defaults to 0 so it could not be lowered. Further passes
    is responsible for materializing the concrete scale when the
    scheme parameter is known.

    Note that `scale` is a 64-bit integer and may not represent the
    large scale like 2^80 or 2^90 in CKKS.

    When further lowered, it could be lowered to bgv.mul_plain
    or ckks.mul_plain depending on the scheme.
  }];

  let arguments = (ins
    AnyType:$input,
    DefaultValuedAttr<I64Attr, "0">:$scale,
    DefaultValuedAttr<F64Attr, "0.0">:$scaleFloat
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";

  let hasCanonicalizer = 1;
}

#endif  // LIB_DIALECT_MGMT_IR_MGMTOPS_TD_
