#ifndef HEIR_INCLUDE_DIALECT_SECRET_IR_SECRETOPS_TD_
#define HEIR_INCLUDE_DIALECT_SECRET_IR_SECRETOPS_TD_

include "SecretDialect.td"
include "SecretTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Secret_Op<string mnemonic, list<Trait> traits = []> :
        Op<Secret_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::secret";
}

def Secret_ConcealOp : Secret_Op<"conceal", [Pure]> {
  let summary = "Convert a non-secret value into a secret";
  let description = [{
    Convert a value to a secret containing the same value.

    This op represents a scheme-agnostic encryption operation, as well as a
    "trivial encryption" operation which is needed for some FHE schemes. This
    op is also useful for type materialization in the dialect conversion
    framework.

    Examples:

    ```
    %Y = secret.conceal %value : i32 -> !secret.secret<i32>
    ```
  }];

  let arguments = (ins AnyType:$cleartext);
  let results = (outs Secret:$output);
  let assemblyFormat = "$cleartext attr-dict `:` type($cleartext) `->` type($output)";

  let builders = [
    // Builder to infer output type from the input type
    OpBuilder<(ins "Value":$cleartext)>
  ];

  // TODO(https://github.com/google/heir/issues/108): add a folder that
  // collapses secret<secret<T>> into secret<T> and removes intermediate
  // conceal ops.
  // let hasFolder = 1;
}

def Secret_RevealOp : Secret_Op<"reveal", [Pure]> {
  let summary = "Convert a secret value into a non-secret";
  let description = [{
    Convert a secret into a non-secret containing the same value.

    This op represents a scheme-agnostic decryption operation. This op is also
    useful for target materialization in the dialect conversion framework.

    Examples:

    ```
    %Y = secret.reveal %secret_value : !secret.secret<i32> -> i32
    ```
  }];

  let arguments = (ins Secret:$input);
  let results = (outs AnyType:$cleartext);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($cleartext)";

  let builders = [
    // Builder to infer output type from the input type
    OpBuilder<(ins "Value":$secret)>
  ];
}


def Secret_YieldOp : Secret_Op<"yield", [Pure, ReturnLike, Terminator]>,
  Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Secret yield operation";
  let description = [{
    `secret.yield` is a special terminator operation for blocks inside regions
    in `secret` generic ops. It returns the cleartext value of the
    corresponding private computation to the immediately enclosing `secret`
    generic op.
  }];
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let hasCustomAssemblyFormat = 1;

  // TODO(https://github.com/google/heir/issues/108): add a verifier that
  // ensures the yield op is inside a generic op, that the return type of the
  // yield op matches the enclosing generic, and any other similar
  // verifications that linalg.generic has that would be applicable here. Maybe
  // there are some generic interfaces that produce these verifiers for free,
  // like SingleBlockImplicitTerminator?
  // let hasVerifier = 1;
}

def Secret_GenericOp : Secret_Op<"generic", [
  IsolatedFromAbove,
  SingleBlock,
  SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "Lift a plaintext computation to operate on secrets.";
  let description = [{
    `secret.generic` lifts a plaintext computation to operate on one or more
    secrets. The lifted computation is represented as a region containing a
    single block terminated by `secret.yield`. The arguments of the `secret.generic`
    may include one or more `!secret.secret` types. The arguments of the block
    in the op's body correspond to the underlying plaintext types of the secrets.

    Basic examples:

    Add two secret integers together

    ```
    %Z = secret.generic ins(%X, %Y : !secret.secret<i32>, !secret.secret<i32>) {
      ^bb0(%x: i32, %y: i32) :
        %z = arith.addi %x, %y: i32
        secret.yield %z : i32
      } -> (!secret.secret<i32>)
    ```

    Add a secret value with a plaintext value. I.e., not all arguments to the
    op need be secret.

    ```
    %Z = secret.generic ins(%X, %Y : i32, !secret.secret<i32>) {
      ^bb0(%x: i32, %y: i32) :
        %z = arith.addi %x, %y: i32
        secret.yield %z : i32
      } -> (!secret.secret<i32>)
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      CArg<"ValueRange">:$inputs,
      CArg<"TypeRange">:$outputTypes,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>">:$bodyBuilder
    )>
  ];

  let extraClassDeclaration = [{
    /// Defining the function type we use for building the body of affine.for.
    using BodyBuilderFn = function_ref<void(OpBuilder &, Location, ValueRange)>;
  }];

  // let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;

  // TODO(https://github.com/google/heir/issues/108): add a folder that removes
  // the op if there are no secret inputs.
  // let hasFolder = 1;

  // TODO(https://github.com/google/heir/issues/108): add a verifier that
  // ensures the arguments to the op match the arguments of the block.
  // let hasVerifier = 1;
}


#endif  // HEIR_INCLUDE_DIALECT_SECRET_IR_SECRETOPS_TD_
