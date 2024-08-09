#ifndef LIB_DIALECT_SECRET_IR_SECRETOPS_TD_
#define LIB_DIALECT_SECRET_IR_SECRETOPS_TD_

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


def Secret_SeparatorOp : Secret_Op<"separator", [MemoryEffects<[MemWrite]>]> {
  let summary = "Convert a non-secret value into a secret";

  let description = [{
    This operation is used as a separation boundary between logical subunits of
    the module. This is used in conjunction with
    `--secret-distribute-generic=distribute-through=secret.separator` to break a
    generic around these separators and allow for optimization passses to
    analyze and optimize the sub-units locally.

    In order to allow bufferization of modules with this operation, we must
    register a (bogus) memory effect that also prevents this operation from
    being trivially dead during operation folding.

    This operation also accepts operands, which act as boundaries between the
    logical units. This enforces separation of memref and affine optimizations
    between the subunits, preventing optimizations from removing the operand and
    combining the two separated regions. The operand can be thought of as an
    return value of the logical subunit.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);

  let assemblyFormat = "attr-dict ($inputs^ `:` type($inputs))?";
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

    ```mlir
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
}

def Secret_RevealOp : Secret_Op<"reveal", [Pure]> {
  let summary = "Convert a secret value into a non-secret";
  let description = [{
    Convert a secret into a non-secret containing the same value.

    This op represents a scheme-agnostic decryption operation. This op is also
    useful for target materialization in the dialect conversion framework.

    Examples:

    ```mlir
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


def Secret_YieldOp : Secret_Op<"yield", [
    Pure, ReturnLike, Terminator, HasParent<"GenericOp">
]>,
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
  let hasVerifier = 1;
}

def Secret_GenericOp : Secret_Op<"generic", [
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

    `secret.generic` is not isolated from above, so you may directly reference
    values in the enclosing scope. This is required to support using
    `secret.generic` inside of ops with `AffineScope`, while having the body
    of the generic use the induction variables defined by the affine scope.

    Basic examples:

    Add two secret integers together

    ```mlir
    %Z = secret.generic ins(%X, %Y : !secret.secret<i32>, !secret.secret<i32>) {
      ^bb0(%x: i32, %y: i32):
        %z = arith.addi %x, %y: i32
        secret.yield %z : i32
      } -> (!secret.secret<i32>)
    ```

    Add a secret value with a plaintext value. I.e., not all arguments to the
    op need be secret.

    ```mlir
    %Z = secret.generic ins(%X, %Y : i32, !secret.secret<i32>) {
      ^bb0(%x: i32, %y: i32):
        %z = arith.addi %x, %y: i32
        secret.yield %z : i32
      } -> (!secret.secret<i32>)
    ```

    The same as above, but the plaintext op is not passed through the basic
    block.

    ```mlir
    %y = arith.constant 7: i32
    %Z = secret.generic ins(%X : !secret.secret<i32>) {
      ^bb0(%x: i32):
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

    // Returns the unique YieldOp in the body of this generic
    YieldOp getYieldOp();

    // Returns the OpOperand of this generic op corresponding to the given
    // value. Returns nullptr if the value is not a block argument for this
    // secret.generic.
    OpOperand *getOpOperandForBlockArgument(Value value);

    // Returns the integer index of a result if the given value is a result of
    // the generic op.
    std::optional<int> findResultIndex(Value value);

    // Clones a generic op and adds new yielded values. Returns the new op and
    // the value range corresponding to the new result values of the generic.
    // Callers can follow this method with something like the following to
    // replace the current generic op with the result of this method. Always
    // adds the new yielded values to the end of the list of yielded values.
    //
    // auto [modifiedGeneric, newResults] =
    //     genericOp.addNewYieldedValues(newResults, rewriter);
    // rewriter.replaceOp(
    //    genericOp,
    //    ValueRange(modifiedGeneric.getResults()
    //                              .drop_back(newResults.size())));
    std::pair<GenericOp, ValueRange> addNewYieldedValues(
      ValueRange newValuesToYield, PatternRewriter &rewriter);

    // Clones the current op with the yielded values in `yieldedValuesToRemove`
    // removed. Users can replace the current op with the result of this method
    // as follows:
    //
    // SmallVector<Value> remainingResults;
    // auto modifiedGeneric =
    //     op.removeYieldedValues(valuesToRemove, rewriter, remainingResults);
    // rewriter.replaceAllUsesWith(remainingResults, modifiedGeneric.getResults());
    // rewriter.eraseOp(op);
    GenericOp removeYieldedValues(
      ValueRange yieldedValuesToRemove,
      PatternRewriter &rewriter,
      SmallVector<Value> &remainingResults);

    // Same as above, but the exact operand indices of the yielded values to
    // remove are specified. Can be useful if you want to use this method to
    // dedupe yielded values.
    GenericOp removeYieldedValues(
      ArrayRef<int> yieldedIndicesToRemove,
      PatternRewriter &rewriter,
      SmallVector<Value> &remainingResults);

    // Modifies a GenericOp in place by taking the given op inside the generic
    // body and lifting it into a new single-op generic before the context
    // generic op. Returns the newly created GenericOp.
    //
    // For extractOpAfterGeneric, see SecretOps.h (it's a non-member function).
    GenericOp extractOpBeforeGeneric(
      Operation *opToExtract, PatternRewriter &rewriter);

    // Inlines the GenericOp in place, dropping any secret types involved.
    // Extra `operands` argument allows a conversion pattern to pass
    // adaptor.getOperands().
    void inlineInPlaceDroppingSecrets(PatternRewriter &rewriter, ValueRange operands);

    void inlineInPlaceDroppingSecrets(PatternRewriter &rewriter) {
      inlineInPlaceDroppingSecrets(rewriter, getOperands());
    }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def CastOp : Secret_Op<"cast", [Pure]> {
  let summary = "A placeholder cast from one secret type to another";
  let description = [{
    A `cast` operation represents a type cast from one secret type to another,
    that is used to enable the intermixing of various equivalent secret types
    before a lower-level FHE scheme has been chosen.

    For example, `secret.cast` can be used to convert a `secret<i8>` to a
    `secret<tensor<8xi1>>` as a compatibility layer between boolean and
    non-boolean parts of a program. The pass that later lowers the IR to
    specific FHE schemes would need to replace these casts with appropriate
    scheme-specific operations, and it is left to those later passes to
    determine which casts are considered valid.

    Example:

    ```mlir
    %result = secret.cast %0 : !secret.secret<i8> to !secret.secret<tensor<8xi1>>
    %result2 = secret.cast %0 : !secret.secret<i8> to !secret.secret<tensor<2xi4>>
    ```
  }];

  let arguments = (ins Secret:$input);
  let results = (outs Secret:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` qualified(type($output))
  }];
  let hasFolder = 1;
}


#endif  // LIB_DIALECT_SECRET_IR_SECRETOPS_TD_
