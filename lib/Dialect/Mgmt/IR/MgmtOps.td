#ifndef LIB_DIALECT_MGMT_IR_MGMTOPS_TD_
#define LIB_DIALECT_MGMT_IR_MGMTOPS_TD_

include "lib/Dialect/Mgmt/IR/MgmtDialect.td"

include "lib/Dialect/HEIRInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Mgmt_Op<string mnemonic, list<Trait> traits = []> :
        Op<Mgmt_Dialect, mnemonic, traits # [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
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

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";

  let hasCanonicalizer = 1;
}

def Mgmt_LevelReduceOp : Mgmt_Op<"level_reduce"> {
  let summary = "Reduce the level of input ciphertext by dropping the last k RNS limbs";

  let description = [{
    This scheme-agonistic operation reduces the ciphertext level
    by k, as specified by the `levelToDrop` attribute.

    If `levelToDrop` is not specified, it defaults to 1.

    Input ciphertext is assumed to be in RNS form when further lowered.

    Later passes may lower this op to `bgv.level_reduce`
    or `ckks.level_reduce` depending on the scheme.
  }];

  let arguments = (ins
    AnyType:$input,
    DefaultValuedAttr<I64Attr, "1">:$levelToDrop
  );
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";

  let hasCanonicalizer = 1;
}

def Mgmt_LevelReduceMinOp : Mgmt_Op<"level_reduce_min"> {
  let summary = "Reduce the level of input ciphertext to the minimum level";
  let description = [{
    This scheme-agonistic operation reduces the ciphertext level
    to the minimum allowable level (typically 1 RNS limb). This is used in
    HALO-style optimizations to ensure loop iter args and inits are invariant
    across loop iterations.
  }];
  let arguments = (ins AnyType:$input);
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

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def Mgmt_BootstrapOp : Mgmt_Op<"bootstrap", [ResetsMulDepthOpInterface]> {
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

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def Mgmt_AdjustScaleOp : Mgmt_Op<"adjust_scale", [IncreasesMulDepthOpInterface]> {
  let summary = "Adjust the scale of the input ciphertext (for BGV and CKKS)";

  let description = [{
    This is scheme-agonistic operation that adjust the scale of the input
    ciphertext. This is an opaque operation, and the concrete value of the
    scale is determined by other methods.

    To distinguish different opaque adjust_scale operations, the `id` attribute
    is used.

    At the time of `secret-insert-mgmt-<scheme>`, the concrete scale
    is not known as the scheme parameter is not generated.
    Further passes like `populate-scale-<scheme>` is responsible for materializing
    the concrete scale when the scheme parameter is known.

    When further lowered, it could be lowered to `bgv.mul_plain`
    or `ckks.mul_plain` depending on the scheme.
  }];

  let arguments = (ins AnyType:$input, I64Attr:$id);
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
  let hasCanonicalizer = 1;
}

def Mgmt_InitOp : Mgmt_Op<"init",
    [MemoryEffects<[MemWrite]>, ElementwiseMappable, SameOperandsAndResultType, ConditionallySpeculatable]> {
  let summary = "Init the plaintext with mgmt attributes";
  let description = [{
    This is a scheme-agnostic operation that initializes the plaintext
    with `mgmt` attributes.

    Plaintext has multiple sources, e.g. function argument, arith.constant,
    tensor.empty, etc. However, they may have multiple uses in HE circuit
    and the level/scale information for them may be different, so we could
    not annotate them with `mgmt` attributes directly, as they could not have
    more than one annotation.

    Also, `mgmt` attributes annotated on them may get lost as other optimization
    like CSE or constant folding may canonicalize them away.

    To address the problem, for each _use_ of the plaintext, we insert an `mgmt.init`
    operation to initialize the plaintext with `mgmt` attributes.

    Technical reasons for registering memory effects:

    Register a (bogus) memory effect to prevent CSE from merging this op.
    Two mgmt.init ops could be seen as equivalent only if they have the same
    MgmtAttr with *level/dimension/scale* annotated, otherwise we could not
    judge whether they are equivalent or not. In practice, we create the op first
    and only in later analyses we know whether they are equivalent or not.

    ConditionallySpeculatable is for isSpeculatable check in hoisting canonicalization.
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";

  let extraClassDeclaration = [{
    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability() {
      return Speculation::Speculatable;
    }
  }];
}

#endif  // LIB_DIALECT_MGMT_IR_MGMTOPS_TD_
