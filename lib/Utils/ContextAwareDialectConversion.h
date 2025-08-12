#ifndef LIB_UTILS_CONTEXTAWAREDIALECTCONVERSION_H_
#define LIB_UTILS_CONTEXTAWAREDIALECTCONVERSION_H_

#include <memory>
#include <optional>

#include "lib/Utils/ContextAwareTypeConversion.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

// HEIR: This file is a port of DialectConversion.h from MLIR upstream, with
// the following changes:
//
// - Replaced uses of TypeConverter with AttributeAwareTypeConverter
// - Changed ConversionPattern to ContextAwareConversionPattern
// - Removed ConversionTarget so it can be reused from upstream
// - Removed reconcileUnrealizedCasts as it can be reused from upstream
// - Removed 1-N dialect conversion hooks, though some support for 1-N
//   conversion remains in the TypeConverter internals.
// - Removed PDL stuff
// - Removed all drivers except applyPartialConversion

class ContextAwareConversionPatternRewriter;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Base class for the conversion patterns. This pattern class enables type
/// conversions, and other uses specific to the conversion framework. As such,
/// patterns of this type can only be used with the 'apply*' methods below.
class ContextAwareConversionPattern : public RewritePattern {
 public:
  /// Hook for derived classes to implement combined matching and rewriting.
  /// This overload supports only 1:1 replacements.
  virtual LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ContextAwareConversionPatternRewriter& rewriter) const {
    llvm_unreachable("matchAndRewrite is not implemented");
  }
  virtual LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<ValueRange> operands,
      ContextAwareConversionPatternRewriter& rewriter) const {
    return matchAndRewrite(op, getOneToOneAdaptorOperands(operands), rewriter);
  }

  /// Attempt to match and rewrite the IR root at the specified operation.
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const final;

  /// Return the type converter held by this pattern, or nullptr if the pattern
  /// does not require type conversion.
  const ContextAwareTypeConverter* getTypeConverter() const {
    return typeConverter;
  }

 protected:
  /// See `RewritePattern::RewritePattern` for information on the other
  /// available constructors.
  using RewritePattern::RewritePattern;
  /// Construct a conversion pattern with the given converter, and forward the
  /// remaining arguments to RewritePattern.
  template <typename... Args>
  ContextAwareConversionPattern(const ContextAwareTypeConverter& typeConverter,
                                Args&&... args)
      : RewritePattern(std::forward<Args>(args)...),
        typeConverter(&typeConverter) {}

  /// Given an array of value ranges, which are the inputs to a 1:N adaptor,
  /// try to extract the single value of each range to construct a the inputs
  /// for a 1:1 adaptor.
  ///
  /// This function produces a fatal error if at least one range has 0 or
  /// more than 1 value: "pattern 'name' does not support 1:N conversion"
  SmallVector<Value> getOneToOneAdaptorOperands(
      ArrayRef<ValueRange> operands) const;

 protected:
  /// An optional type converter for use by this pattern.
  const ContextAwareTypeConverter* typeConverter = nullptr;
};

/// ContextAwareOpConversionPattern is a wrapper around
/// ContextAwareConversionPattern that allows for matching and rewriting
/// against an instance of a derived operation class as opposed to a raw
/// Operation.
template <typename SourceOp>
class ContextAwareOpConversionPattern : public ContextAwareConversionPattern {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;
  using ContextAwareConversionPattern::matchAndRewrite;

  ContextAwareOpConversionPattern(MLIRContext* context,
                                  PatternBenefit benefit = 1)
      : ContextAwareConversionPattern(SourceOp::getOperationName(), benefit,
                                      context) {}
  ContextAwareOpConversionPattern(
      const ContextAwareTypeConverter& typeConverter, MLIRContext* context,
      PatternBenefit benefit = 1)
      : ContextAwareConversionPattern(
            typeConverter, SourceOp::getOperationName(), benefit, context) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ContextAwareConversionPatternRewriter& rewriter) const final {
    auto sourceOp = cast<SourceOp>(op);
    return matchAndRewrite(sourceOp, OpAdaptor(operands, sourceOp), rewriter);
  }

  virtual LogicalResult matchAndRewrite(
      SourceOp op, OpAdaptor adaptor,
      ContextAwareConversionPatternRewriter& rewriter) const {
    llvm_unreachable("matchAndRewrite is not implemented");
  }
};

//===----------------------------------------------------------------------===//
// Conversion PatternRewriter
//===----------------------------------------------------------------------===//

namespace detail {
struct ContextAwareConversionPatternRewriterImpl;
}  // namespace detail

/// This class implements a pattern rewriter for use with
/// ContextAwareConversionPatterns. It extends the base PatternRewriter and
/// provides special conversion specific hooks.
class ContextAwareConversionPatternRewriter final : public PatternRewriter {
 public:
  ~ContextAwareConversionPatternRewriter() override;

  /// Apply a signature conversion to given block. This replaces the block with
  /// a new block containing the updated signature. The operations of the given
  /// block are inlined into the newly-created block, which is returned.
  ///
  /// If no block argument types are changing, the original block will be
  /// left in place and returned.
  ///
  /// A signature conversion must be provided. (Type converters can construct
  /// a signature conversion with `convertBlockSignature`.)
  ///
  /// Optionally, a type converter can be provided to build materializations.
  /// Note: If no type converter was provided or the type converter does not
  /// specify any suitable argument/target materialization rules, the dialect
  /// conversion may fail to legalize unresolved materializations.
  Block* applySignatureConversion(
      Block* block, ContextAwareTypeConverter::SignatureConversion& conversion,
      const ContextAwareTypeConverter* converter = nullptr);

  /// Apply a signature conversion to each block in the given region. This
  /// replaces each block with a new block containing the updated signature. If
  /// an updated signature would match the current signature, the respective
  /// block is left in place as is. (See `applySignatureConversion` for
  /// details.) The new entry block of the region is returned.
  ///
  /// SignatureConversions are computed with the specified type converter.
  /// This function returns "failure" if the type converter failed to compute
  /// a SignatureConversion for at least one block.
  ///
  /// Optionally, a special SignatureConversion can be specified for the entry
  /// block. This is because the types of the entry block arguments are often
  /// tied semantically to the operation.
  FailureOr<Block*> convertRegionTypes(
      Region* region, const ContextAwareTypeConverter& converter,
      ContextAwareTypeConverter::SignatureConversion* entryConversion =
          nullptr);

  /// Replace all the uses of the block argument `from` with value `to`.
  void replaceUsesOfBlockArgument(BlockArgument from, Value to);

  /// Return the converted value of 'key' with a type defined by the type
  /// converter of the currently executing pattern. Return nullptr in the case
  /// of failure, the remapped value otherwise.
  Value getRemappedValue(Value key);

  /// Return the converted values that replace 'keys' with types defined by the
  /// type converter of the currently executing pattern. Returns failure if the
  /// remap failed, success otherwise.
  LogicalResult getRemappedValues(ValueRange keys,
                                  SmallVectorImpl<Value>& results);

  //===--------------------------------------------------------------------===//
  // PatternRewriter Hooks
  //===--------------------------------------------------------------------===//

  /// Indicate that the conversion rewriter can recover from rewrite failure.
  /// Recovery is supported via rollback, allowing for continued processing of
  /// patterns even if a failure is encountered during the rewrite step.
  bool canRecoverFromRewriteFailure() const override { return true; }

  /// Replace the given operation with the new values. The number of op results
  /// and replacement values must match. The types may differ: the dialect
  /// conversion driver will reconcile any surviving type mismatches at the end
  /// of the conversion process with source materializations. The given
  /// operation is erased.
  void replaceOp(Operation* op, ValueRange newValues) override;

  /// Replace the given operation with the results of the new op. The number of
  /// op results must match. The types may differ: the dialect conversion
  /// driver will reconcile any surviving type mismatches at the end of the
  /// conversion process with source materializations. The original operation
  /// is erased.
  void replaceOp(Operation* op, Operation* newOp) override;

  /// Replace the given operation with the new value ranges. The number of op
  /// results and value ranges must match. The given  operation is erased.
  void replaceOpWithMultiple(Operation* op, ArrayRef<ValueRange> newValues);

  /// PatternRewriter hook for erasing a dead operation. The uses of this
  /// operation *must* be made dead by the end of the conversion process,
  /// otherwise an assert will be issued.
  void eraseOp(Operation* op) override;

  /// PatternRewriter hook for erase all operations in a block. This is not yet
  /// implemented for dialect conversion.
  void eraseBlock(Block* block) override;

  /// PatternRewriter hook for inlining the ops of a block into another block.
  void inlineBlockBefore(Block* source, Block* dest, Block::iterator before,
                         ValueRange argValues = std::nullopt) override;
  using PatternRewriter::inlineBlockBefore;

  /// PatternRewriter hook for updating the given operation in-place.
  /// Note: These methods only track updates to the given operation itself,
  /// and not nested regions. Updates to regions will still require notification
  /// through other more specific hooks above.
  void startOpModification(Operation* op) override;

  /// PatternRewriter hook for updating the given operation in-place.
  void finalizeOpModification(Operation* op) override;

  /// PatternRewriter hook for updating the given operation in-place.
  void cancelOpModification(Operation* op) override;

  /// Return a reference to the internal implementation.
  detail::ContextAwareConversionPatternRewriterImpl& getImpl();

 private:
  // Allow OperationConverter to construct new rewriters.
  friend struct OperationConverter;

  /// Conversion pattern rewriters must not be used outside of dialect
  /// conversions. They apply some IR rewrites in a delayed fashion and could
  /// bring the IR into an inconsistent state when used standalone.
  explicit ContextAwareConversionPatternRewriter(
      MLIRContext* ctx, const ConversionConfig& config);

  // Hide unsupported pattern rewriter API.
  using OpBuilder::setListener;

  std::unique_ptr<detail::ContextAwareConversionPatternRewriterImpl> impl;
};

//===----------------------------------------------------------------------===//
// Op Conversion Entry Points
//===----------------------------------------------------------------------===//

/// Apply a partial conversion on the given operations and all nested
/// operations. This method converts as many operations to the target as
/// possible, ignoring operations that failed to legalize. This method only
/// returns failure if there ops explicitly marked as illegal.
LogicalResult applyContextAwarePartialConversion(
    ArrayRef<Operation*> ops, const ConversionTarget& target,
    const FrozenRewritePatternSet& patterns,
    ConversionConfig config = ConversionConfig());
LogicalResult applyContextAwarePartialConversion(
    Operation* op, const ConversionTarget& target,
    const FrozenRewritePatternSet& patterns,
    ConversionConfig config = ConversionConfig());

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CONTEXTAWAREDIALECTCONVERSION_H_
