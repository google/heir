
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"

// When using DRR to define new artih ops, one must include an attribute
// describing the overflow semantics. When the new arith op is built from an
// existing op, one should match the overflow attribute and propagate it, but
// when the op is built from scratch, one needs to pass an overflow choice, and
// if there is no reason to pick a particular overflow type, use this helper as
// a default.
defvar DefOverflow = ConstantEnumCase<Arith_IntegerOverflowAttr, "none">;

// Same as `DevOverflow` but for the fastmath setting required by float ops:
defvar DefFastMath = ConstantEnumCase<Arith_FastMathAttr, "none">;

// When constructing a new variadic op, and you want to pass it a single argument
// as its variadic input, use this helper. To understand why this is needed, see
// https://discourse.llvm.org/t/compilation-failure-with-drr-generated-pattern/77385
def MakeSingleResultVariadic: NativeCodeCall<"{ $0 }">;

// Determine if an input integer attr is zero
def IsZeroIntAttr :
    Constraint<
      CPred<"llvm::cast<mlir::IntegerAttr>($0).getValue().isZero()">>;

// Construct an index.cast op, which is necessary here because the default
// builder is not compatible with DRR as of 2024-03-11.
def CreateIndexCastOp : NativeCodeCall<
      "arith::IndexCastOp::create($_builder, $0.getLoc(), $1.getType(), $0)">;

// Returns true if the given value has a single use.
def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;
