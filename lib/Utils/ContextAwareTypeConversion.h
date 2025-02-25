#ifndef LIB_UTILS_CONTEXTAWARETYPECONVERSION_H_
#define LIB_UTILS_CONTEXTAWARETYPECONVERSION_H_

#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

// A class to manage type conversions when the context of the value matters.
// Note this excludes the ability to use this to convert ops that don't define
// values, such as a func.func declaration (isDeclaration() is true).
//
// This framework only supports 1-1 type conversions, without dropping or
// inserting additional types during the conversion.
//
// The inheritance from TypeConverter does nothing, but is necessary to allow
// this class to be used with the DialectConversion framework, particularly
// the ConversionPattern constructor requires a TypeConverter instance.
struct ContextAwareTypeConverter : TypeConverter {
 public:
  ContextAwareTypeConverter() {
    // Required to conform to dialect conversion, otherwise using this in a
    // conversion pattern will always fail.
    addConversion([](Type type) { return type; });
  }

  // Convert a range of values, with converted types stored in newTypes.
  virtual LogicalResult convertValueRangeTypes(
      ValueRange values, SmallVectorImpl<Type> &newTypes) const = 0;

  // Convert types of the arguments and results of a function.
  virtual LogicalResult convertFuncSignature(
      FunctionOpInterface funcOp, SmallVectorImpl<Type> &newArgTypes,
      SmallVectorImpl<Type> &newResultTypes) const = 0;

  // For use with the normal DialectConversion framework to trigger conversion
  // via dynamic legality checks.
  bool isLegal(Operation *op) const;
  bool isLegal(FunctionOpInterface funcOp) const;
  bool isLegal(func::FuncOp funcOp) const {
    return isLegal(cast<FunctionOpInterface>(funcOp.getOperation()));
  }
};

// A ContextAwareTypeConverter for which the only context needed is an
// attribute, which this class is in charge of retrieving.
struct AttributeAwareTypeConverter : ContextAwareTypeConverter {
 public:
  virtual FailureOr<Type> convert(Type type, Attribute attr) const = 0;

  // Return an Attribute used by convertValueRangeTypes to convert the type of
  // the input `value`. If no usable attribute is found, returns a failure.
  // This may indicate that no type conversion is necessary. As a result,
  // the returned Attribute is never nullptr.
  virtual FailureOr<Attribute> getContextualAttr(Value value) const = 0;

  // Convert a range of values, with converted types stored in newTypes.
  LogicalResult convertValueRangeTypes(
      ValueRange values, SmallVectorImpl<Type> &newTypes) const override;

  // Convert types of the arguments and results of a function.
  LogicalResult convertFuncSignature(
      FunctionOpInterface funcOp, SmallVectorImpl<Type> &newArgTypes,
      SmallVectorImpl<Type> &newResultTypes) const override;
};

// An AttributeAwareTypeConverter for which the attribute is determined uniquely
// by a specific string name on the defining op or as a func arg attr.
struct UniquelyNamedAttributeAwareTypeConverter : AttributeAwareTypeConverter {
 public:
  UniquelyNamedAttributeAwareTypeConverter(StringRef attrName)
      : attrName(attrName) {}

  FailureOr<Attribute> getContextualAttr(Value value) const override {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto *parentOp = blockArg.getOwner()->getParentOp();
      auto funcOp = dyn_cast<FunctionOpInterface>(parentOp);
      if (!funcOp) return failure();
      auto argAttr = funcOp.getArgAttr(blockArg.getArgNumber(), attrName);
      if (!argAttr) return failure();

      return argAttr;
    }

    auto *parentOp = value.getDefiningOp();
    if (!parentOp || !parentOp->hasAttr(attrName)) return failure();

    return parentOp->getAttr(attrName);
  }

 private:
  std::string attrName;
};

struct ConvertFuncWithContextAwareTypeConverter
    : public OpRewritePattern<func::FuncOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  ConvertFuncWithContextAwareTypeConverter(
      const ContextAwareTypeConverter &contextAwareTypeConverter,
      MLIRContext *context)
      : OpRewritePattern(context),
        contextAwareTypeConverter(&contextAwareTypeConverter) {}

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override;

  // An overridable hook that allows subclasses to perform additional
  // modifications of the func op after its type signature has been converted.
  // For example, a subclass may use this hook to modify arg attrs.
  virtual LogicalResult finalizeFuncOpModification(
      func::FuncOp op, ArrayRef<Type> oldArgTypes,
      ArrayRef<Type> oldResultTypes, PatternRewriter &rewriter) const {
    return success();
  };

 private:
  const ContextAwareTypeConverter *contextAwareTypeConverter;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CONTEXTAWARETYPECONVERSION_H_
