#ifndef LIB_UTILS_CONTEXTAWARETYPECONVERSION_H_
#define LIB_UTILS_CONTEXTAWARETYPECONVERSION_H_

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "lib/Utils/AttributeUtils.h"
#include "llvm/include/llvm/ADT/DenseMapInfo.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/Hashing.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/PointerIntPair.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"       // from @llvm-project
#include "llvm/include/llvm/Support/RWMutex.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"         // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"       // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {

// We simplify the Context for "context aware" is always a single attribute
struct TypeAndAttribute {
  Type type;
  Attribute attr;
};

// A class to manage type conversions when the context of the value matters.
// Note this excludes the ability to use this to convert ops that don't define
// values, such as a func.func declaration (isDeclaration() is true).
class ContextAwareTypeConverter {
 public:
  virtual ~ContextAwareTypeConverter() = default;
  ContextAwareTypeConverter() = default;
  // Copy the registered conversions, but not the caches
  ContextAwareTypeConverter(const ContextAwareTypeConverter& other)
      : conversions(other.conversions),
        argumentMaterializations(other.argumentMaterializations),
        sourceMaterializations(other.sourceMaterializations),
        targetMaterializations(other.targetMaterializations),
        typeAttributeConversions(other.typeAttributeConversions) {}
  ContextAwareTypeConverter& operator=(const ContextAwareTypeConverter& other) {
    conversions = other.conversions;
    argumentMaterializations = other.argumentMaterializations;
    sourceMaterializations = other.sourceMaterializations;
    targetMaterializations = other.targetMaterializations;
    typeAttributeConversions = other.typeAttributeConversions;
    return *this;
  }

  // HEIR: the main way to determine
  // the input `value`. If no usable attribute is found, returns a failure.
  // This may indicate that no type conversion is necessary. As a result, the
  // returned Attribute is never nullptr.
  virtual FailureOr<Attribute> getContextualAttr(Value value) const = 0;

  // HEIR: FuncOps that are declarations don't have SSA values to use for
  // context, so the subclass of this type converter must define how to handle
  // function signatures.
  virtual LogicalResult convertFuncSignature(
      FunctionOpInterface funcOp, SmallVectorImpl<Type>& newArgTypes,
      SmallVectorImpl<Type>& newResultTypes) const = 0;

  /// This class provides all of the information necessary to convert a type
  /// signature.
  class SignatureConversion {
   public:
    SignatureConversion(unsigned numOrigInputs)
        : remappedInputs(numOrigInputs) {}

    /// This struct represents a range of new types or a single value that
    /// remaps an existing signature input.
    struct InputMapping {
      size_t inputNo, size;
      Value replacementValue;
    };

    /// Return the argument types for the new signature.
    ArrayRef<Type> getConvertedTypes() const { return argTypes; }

    /// Get the input mapping for the given argument.
    std::optional<InputMapping> getInputMapping(unsigned input) const {
      return remappedInputs[input];
    }

    //===------------------------------------------------------------------===//
    // Conversion Hooks
    //===------------------------------------------------------------------===//

    /// Remap an input of the original signature with a new set of types. The
    /// new types are appended to the new signature conversion.
    void addInputs(unsigned origInputNo, ArrayRef<Type> types);

    /// Append new input types to the signature conversion, this should only be
    /// used if the new types are not intended to remap an existing input.
    void addInputs(ArrayRef<Type> types);

    /// Remap an input of the original signature to another `replacement`
    /// value. This drops the original argument.
    void remapInput(unsigned origInputNo, Value replacement);

   private:
    /// Remap an input of the original signature with a range of types in the
    /// new signature.
    void remapInput(unsigned origInputNo, unsigned newInputNo,
                    unsigned newInputCount = 1);

    /// The remapping information for each of the original arguments.
    SmallVector<std::optional<InputMapping>, 4> remappedInputs;

    /// The set of new argument types.
    SmallVector<Type, 4> argTypes;
  };

  /// The general result of a type attribute conversion callback, allowing
  /// for early termination. The default constructor creates the na case.
  class AttributeConversionResult {
   public:
    constexpr AttributeConversionResult() : impl() {}
    AttributeConversionResult(Attribute attr) : impl(attr, resultTag) {}

    static AttributeConversionResult result(Attribute attr);
    static AttributeConversionResult na();
    static AttributeConversionResult abort();

    bool hasResult() const;
    bool isNa() const;
    bool isAbort() const;

    Attribute getResult() const;

   private:
    AttributeConversionResult(Attribute attr, unsigned tag) : impl(attr, tag) {}

    llvm::PointerIntPair<Attribute, 2> impl;
    // Note that na is 0 so that we can use PointerIntPair's default
    // constructor.
    static constexpr unsigned naTag = 0;
    static constexpr unsigned resultTag = 1;
    static constexpr unsigned abortTag = 2;
  };

  /// Register a conversion function. A conversion function must be convertible
  /// to any of the following forms (where `T` is a class derived from `Type`):
  ///
  ///   * std::optional<Type>(T, Value)
  ///     - This form represents a 1-1 type conversion. It should return nullptr
  ///       or `std::nullopt` to signify failure. If `std::nullopt` is returned,
  ///       the converter is allowed to try another conversion function to
  ///       perform the conversion.
  ///   * std::optional<LogicalResult>(T, ValueRange, SmallVectorImpl<Type> &)
  ///     - This form represents a 1-N type conversion. It should return
  ///       `failure` or `std::nullopt` to signify a failed conversion. If the
  ///       new set of types is empty, the type is removed and any usages of the
  ///       existing value are expected to be removed during conversion. If
  ///       `std::nullopt` is returned, the converter is allowed to try another
  ///       conversion function to perform the conversion.
  ///
  /// Note: When attempting to convert a type, e.g. via 'convertType', the
  ///       mostly recently added conversions will be invoked first.
  template <
      typename FnT,
      // Type
      typename T =
          typename llvm::function_traits<std::decay_t<FnT>>::template arg_t<0>,
      // Attribute
      typename A =
          typename llvm::function_traits<std::decay_t<FnT>>::template arg_t<1>>
  void addConversion(FnT&& callback) {
    registerConversion(wrapCallback<T, A>(std::forward<FnT>(callback)));
  }

  /// All of the following materializations require function objects that are
  /// convertible to the following form:
  ///   `Value(OpBuilder &, T, ValueRange, Location)`,
  /// where `T` is any subclass of `Type`. This function is responsible for
  /// creating an operation, using the OpBuilder and Location provided, that
  /// "casts" a range of values into a single value of the given type `T`. It
  /// must return a Value of the type `T` on success and `nullptr` if
  /// it failed but other materialization should be attempted. Materialization
  /// functions must be provided when a type conversion may persist after the
  /// conversion has finished.
  ///
  /// Note: Target materializations may optionally accept an additional Type
  /// parameter, which is the original type of the SSA value. Furthermore, `T`
  /// can be a TypeRange; in that case, the function must return a
  /// SmallVector<Value>.

  /// This method registers a materialization that will be called when
  /// converting (potentially multiple) block arguments that were the result of
  /// a signature conversion of a single block argument, to a single SSA value
  /// with the old block argument type.
  ///
  /// Note: Argument materializations are used only with the 1:N dialect
  /// conversion driver. The 1:N dialect conversion driver will be removed soon
  /// and so will be argument materializations.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<1>>
  void addArgumentMaterialization(FnT&& callback) {
    argumentMaterializations.emplace_back(
        wrapMaterialization<T>(std::forward<FnT>(callback)));
  }

  /// This method registers a materialization that will be called when
  /// converting a replacement value back to its original source type.
  /// This is used when some uses of the original value persist beyond the main
  /// conversion.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<1>>
  void addSourceMaterialization(FnT&& callback) {
    sourceMaterializations.emplace_back(
        wrapMaterialization<T>(std::forward<FnT>(callback)));
  }

  /// This method registers a materialization that will be called when
  /// converting a value to a target type according to a pattern's type
  /// converter.
  ///
  /// Note: Target materializations can optionally inspect the "original"
  /// type. This type may be different from the type of the input value.
  /// For example, let's assume that a conversion pattern "P1" replaced an SSA
  /// value "v1" (type "t1") with "v2" (type "t2"). Then a different conversion
  /// pattern "P2" matches an op that has "v1" as an operand. Let's furthermore
  /// assume that "P2" determines that the converted target type of "t1" is
  /// "t3", which may be different from "t2". In this example, the target
  /// materialization will be invoked with: outputType = "t3", inputs = "v2",
  /// originalType = "t1". Note that the original type "t1" cannot be recovered
  /// from just "t3" and "v2"; that's why the originalType parameter exists.
  ///
  /// Note: During a 1:N conversion, the result types can be a TypeRange. In
  /// that case the materialization produces a SmallVector<Value>.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<1>>
  void addTargetMaterialization(FnT&& callback) {
    targetMaterializations.emplace_back(
        wrapTargetMaterialization<T>(std::forward<FnT>(callback)));
  }

  /// Register a conversion function for attributes within types. Type
  /// converters may call this function in order to allow hoking into the
  /// translation of attributes that exist within types. For example, a type
  /// converter for the `memref` type could use these conversions to convert
  /// memory spaces or layouts in an extensible way.
  ///
  /// The conversion functions take a non-null Type or subclass of Type and a
  /// non-null Attribute (or subclass of Attribute), and returns a
  /// `AttributeConversionResult`. This result can either contain an
  /// `Attribute`, which may be `nullptr`, representing the conversion's
  /// success, `AttributeConversionResult::na()` (the default empty value),
  /// indicating that the conversion function did not apply and that further
  /// conversion functions should be checked, or
  /// `AttributeConversionResult::abort()` indicating that the conversion
  /// process should be aborted.
  ///
  /// Registered conversion functions are called in the reverse of the order in
  /// which they were registered.
  template <
      typename FnT,
      typename T =
          typename llvm::function_traits<std::decay_t<FnT>>::template arg_t<0>,
      typename A =
          typename llvm::function_traits<std::decay_t<FnT>>::template arg_t<1>>
  void addTypeAttributeConversion(FnT&& callback) {
    registerTypeAttributeConversion(
        wrapTypeAttributeConversion<T, A>(std::forward<FnT>(callback)));
  }

  /// Convert the given type. This function should return failure if no valid
  /// conversion exists, success otherwise. If the new set of types is empty,
  /// the type is removed and any usages of the existing value are expected to
  /// be removed during conversion.
  ///
  /// HEIR: the added argument Attribute corresponds to the context of the type
  LogicalResult convertType(Type t, Attribute attr,
                            SmallVectorImpl<Type>& results) const;
  /// Here the value is used as context
  LogicalResult convertType(Type t, Value v,
                            SmallVectorImpl<Type>& results) const;

  /// This hook simplifies defining 1-1 type conversions. This function returns
  /// the type to convert to on success, and a null type on failure.
  ///
  /// HEIR: the added argument Attribute corresponds to the context of the type
  Type convertType(Type t, Attribute attr) const;

  /// Attempts a 1-1 type conversion, expecting the result type to be
  /// `TargetType`. Returns the converted type cast to `TargetType` on success,
  /// and a null type on conversion or cast failure.
  ///
  /// HEIR: the added argument Value v corresponds to the context of the type
  template <typename TargetType>
  TargetType convertType(Type t, Attribute attr) const {
    return dyn_cast_or_null<TargetType>(convertType(t, attr));
  }

  /// Convert the given set of types, filling 'results' as necessary. This
  /// returns failure if the conversion of any of the types fails, success
  /// otherwise.
  ///
  /// HEIR: the added argument array of attributes corresponds to the context of
  /// each type
  LogicalResult convertTypes(TypeRange types, ArrayRef<Attribute> attributes,
                             SmallVectorImpl<Type>& results) const;
  LogicalResult convertTypes(TypeRange types, ValueRange values,
                             SmallVectorImpl<Type>& results) const;

  /// Return true if the given type is legal for this type converter, i.e. the
  /// type converts to itself.
  ///
  /// HEIR: the added argument Attribute corresponds to the context of the type
  bool isLegal(Type type, Attribute attr) const;

  /// Returns true if the function signature is legal.
  ///
  /// HEIR: requires the op for context
  bool isSignatureLegal(FunctionOpInterface funcOp) const;

  /// Same as isLegal, but for type ranges and value ranges
  ///
  /// HEIR: the added argument array of attributes corresponds to the context of
  /// the types in the TypeRange
  bool isLegal(TypeRange types, ArrayRef<Attribute> attributes) const;

  /// Return true if the given operation has legal operand and result types.
  bool isLegal(Operation* op) const;

  /// Return true if the types of block arguments within the region are legal.
  bool isLegal(Region* region) const;

  /// This function converts the type signature of the given block, by invoking
  /// 'convertSignatureArg' for each argument. This function should return a
  /// valid conversion for the signature on success, std::nullopt otherwise.
  std::optional<SignatureConversion> convertBlockSignature(Block* block) const;

  /// Materialize a conversion from a set of types into one result type by
  /// generating a cast sequence of some kind. See the respective
  /// `add*Materialization` for more information on the context for these
  /// methods.
  Value materializeSourceConversion(OpBuilder& builder, Location loc,
                                    Type resultType, ValueRange inputs) const;
  Value materializeTargetConversion(OpBuilder& builder, Location loc,
                                    Type resultType, ValueRange inputs,
                                    Type originalType = {}) const;
  SmallVector<Value> materializeTargetConversion(OpBuilder& builder,
                                                 Location loc,
                                                 TypeRange resultType,
                                                 ValueRange inputs,
                                                 Type originalType = {}) const;

  /// Convert an attribute present `attr` from within the type `type` using
  /// the registered conversion functions. If no applicable conversion has been
  /// registered, return std::nullopt. Note that the empty attribute/`nullptr`
  /// is a valid return value for this function.
  std::optional<Attribute> convertTypeAttribute(Type type,
                                                Attribute attr) const;

 private:
  /// The signature of the callback used to convert a type. If the new set of
  /// types is empty, the type is removed and any usages of the existing value
  /// are expected to be removed during conversion.
  using ConversionCallbackFn = std::function<std::optional<LogicalResult>(
      Type, Attribute, SmallVectorImpl<Type>&)>;

  /// The signature of the callback used to materialize a source/argument
  /// conversion.
  ///
  /// Arguments: builder, result type, inputs, location
  using MaterializationCallbackFn =
      std::function<Value(OpBuilder&, Type, ValueRange, Location)>;

  /// The signature of the callback used to materialize a target conversion.
  ///
  /// Arguments: builder, result types, inputs, location, original type
  using TargetMaterializationCallbackFn = std::function<SmallVector<Value>(
      OpBuilder&, TypeRange, ValueRange, Location, Type)>;

  /// The signature of the callback used to convert a type attribute.
  using TypeAttributeConversionCallbackFn =
      std::function<AttributeConversionResult(Type, Attribute)>;

  /// Generate a wrapper for the given callback. This allows for accepting
  /// different callback forms, that all compose into a single version.
  /// With callback of form: `std::optional<Type>(T, A)`
  template <typename T, typename A, typename FnT>
  std::enable_if_t<std::is_invocable_v<FnT, T, A>, ConversionCallbackFn>
  wrapCallback(FnT&& callback) const {
    return wrapCallback<T, A>(
        [callback = std::forward<FnT>(callback)](
            T type, A attr, SmallVectorImpl<Type>& results) {
          if (std::optional<Type> resultOpt = callback(type, attr)) {
            bool wasSuccess = static_cast<bool>(*resultOpt);
            if (wasSuccess) results.push_back(*resultOpt);
            return std::optional<LogicalResult>(success(wasSuccess));
          }
          return std::optional<LogicalResult>();
        });
  }
  /// With callback of form: `std::optional<LogicalResult>(
  ///     T, A, SmallVectorImpl<Type> &, ArrayRef<Type>)`.
  template <typename T, typename A, typename FnT>
  std::enable_if_t<std::is_invocable_v<FnT, T, A, SmallVectorImpl<Type>&>,
                   ConversionCallbackFn>
  wrapCallback(FnT&& callback) const {
    return [callback = std::forward<FnT>(callback)](
               Type type, Attribute attr,
               SmallVectorImpl<Type>& results) -> std::optional<LogicalResult> {
      T derivedType = dyn_cast<T>(type);
      A derivedAttr = dyn_cast<A>(attr);
      if (!derivedType || !derivedAttr) return std::nullopt;
      return callback(derivedType, derivedAttr, results);
    };
  }

  /// Register a type conversion.
  void registerConversion(ConversionCallbackFn callback) {
    conversions.emplace_back(std::move(callback));
    cachedDirectConversions.clear();
    cachedMultiConversions.clear();
  }

  /// Generate a wrapper for the given argument/source materialization
  /// callback. The callback may take any subclass of `Type` and the
  /// wrapper will check for the target type to be of the expected class
  /// before calling the callback.
  template <typename T, typename FnT>
  MaterializationCallbackFn wrapMaterialization(FnT&& callback) const {
    return [callback = std::forward<FnT>(callback)](
               OpBuilder& builder, Type resultType, ValueRange inputs,
               Location loc) -> Value {
      if (T derivedType = dyn_cast<T>(resultType))
        return callback(builder, derivedType, inputs, loc);
      return Value();
    };
  }

  /// Generate a wrapper for the given target materialization callback.
  /// The callback may take any subclass of `Type` and the wrapper will check
  /// for the target type to be of the expected class before calling the
  /// callback.
  ///
  /// With callback of form:
  /// - Value(OpBuilder &, T, ValueRange, Location, Type)
  /// - SmallVector<Value>(OpBuilder &, TypeRange, ValueRange, Location, Type)
  template <typename T, typename FnT>
  std::enable_if_t<
      std::is_invocable_v<FnT, OpBuilder&, T, ValueRange, Location, Type>,
      TargetMaterializationCallbackFn>
  wrapTargetMaterialization(FnT&& callback) const {
    return [callback = std::forward<FnT>(callback)](
               OpBuilder& builder, TypeRange resultTypes, ValueRange inputs,
               Location loc, Type originalType) -> SmallVector<Value> {
      SmallVector<Value> result;
      if constexpr (std::is_same<T, TypeRange>::value) {
        // This is a 1:N target materialization. Return the produces values
        // directly.
        result = callback(builder, resultTypes, inputs, loc, originalType);
      } else if constexpr (std::is_assignable<Type, T>::value) {
        // This is a 1:1 target materialization. Invoke the callback only if a
        // single SSA value is requested.
        if (resultTypes.size() == 1) {
          // Invoke the callback only if the type class of the callback matches
          // the requested result type.
          if (T derivedType = dyn_cast<T>(resultTypes.front())) {
            // 1:1 materializations produce single values, but we store 1:N
            // target materialization functions in the type converter. Wrap the
            // result value in a SmallVector<Value>.
            Value val =
                callback(builder, derivedType, inputs, loc, originalType);
            if (val) result.push_back(val);
          }
        }
      } else {
        static_assert(sizeof(T) == 0, "T must be a Type or a TypeRange");
      }
      return result;
    };
  }
  /// With callback of form:
  /// - Value(OpBuilder &, T, ValueRange, Location)
  /// - SmallVector<Value>(OpBuilder &, TypeRange, ValueRange, Location)
  template <typename T, typename FnT>
  std::enable_if_t<
      std::is_invocable_v<FnT, OpBuilder&, T, ValueRange, Location>,
      TargetMaterializationCallbackFn>
  wrapTargetMaterialization(FnT&& callback) const {
    return wrapTargetMaterialization<T>(
        [callback = std::forward<FnT>(callback)](
            OpBuilder& builder, T resultTypes, ValueRange inputs, Location loc,
            Type originalType) {
          return callback(builder, resultTypes, inputs, loc);
        });
  }

  /// Generate a wrapper for the given memory space conversion callback. The
  /// callback may take any subclass of `Attribute` and the wrapper will check
  /// for the target attribute to be of the expected class before calling the
  /// callback.
  template <typename T, typename A, typename FnT>
  TypeAttributeConversionCallbackFn wrapTypeAttributeConversion(
      FnT&& callback) const {
    return [callback = std::forward<FnT>(callback)](
               Type type, Attribute attr) -> AttributeConversionResult {
      if (T derivedType = dyn_cast<T>(type)) {
        if (A derivedAttr = dyn_cast_or_null<A>(attr))
          return callback(derivedType, derivedAttr);
      }
      return AttributeConversionResult::na();
    };
  }

  /// Register a memory space conversion, clearing caches.
  void registerTypeAttributeConversion(
      TypeAttributeConversionCallbackFn callback) {
    typeAttributeConversions.emplace_back(std::move(callback));
    // Clear type conversions in case a memory space is lingering inside.
    cachedDirectConversions.clear();
    cachedMultiConversions.clear();
  }

  /// The set of registered conversion functions.
  SmallVector<ConversionCallbackFn, 4> conversions;

  /// The list of registered materialization functions.
  SmallVector<MaterializationCallbackFn, 2> argumentMaterializations;
  SmallVector<MaterializationCallbackFn, 2> sourceMaterializations;
  SmallVector<TargetMaterializationCallbackFn, 2> targetMaterializations;

  /// The list of registered type attribute conversion functions.
  SmallVector<TypeAttributeConversionCallbackFn, 2> typeAttributeConversions;

  /// A set of cached conversions to avoid recomputing in the common case.
  /// Direct 1-1 conversions are the most common, so this cache stores the
  /// successful 1-1 conversions as well as all failed conversions.
  mutable DenseMap<TypeAndAttribute, Type> cachedDirectConversions;
  /// This cache stores the successful 1->N conversions, where N != 1.
  mutable DenseMap<TypeAndAttribute, SmallVector<Type, 2>>
      cachedMultiConversions;
  /// A mutex used for cache access
  mutable llvm::sys::SmartRWMutex<true> cacheMutex;
};

// An AttributeAwareTypeConverter for which the attribute is determined uniquely
// by a specific string name on the defining op or as a func arg attr.
struct UniquelyNamedAttributeAwareTypeConverter : ContextAwareTypeConverter {
  using ContextAwareTypeConverter::ContextAwareTypeConverter;

 public:
  UniquelyNamedAttributeAwareTypeConverter(StringRef attrName)
      : attrName(attrName) {}

  FailureOr<Attribute> getContextualAttr(Value value) const override {
    return findAttributeAssociatedWith(value, attrName);
  }

  LogicalResult convertFuncSignature(
      FunctionOpInterface funcOp, SmallVectorImpl<Type>& newArgTypes,
      SmallVectorImpl<Type>& newResultTypes) const override {
    for (int i = 0; i < funcOp.getNumArguments(); ++i) {
      auto argType = funcOp.getArgumentTypes()[i];
      auto contextAttr = funcOp.getArgAttr(i, attrName);
      if (!contextAttr) {
        newArgTypes.push_back(argType);
        continue;
      }

      auto convertedType = convertType(argType, contextAttr);
      if (convertedType == nullptr) return failure();
      newArgTypes.push_back(convertedType);
    }

    for (int i = 0; i < funcOp.getNumResults(); ++i) {
      auto resultType = funcOp.getResultTypes()[i];
      auto contextAttr = funcOp.getResultAttr(i, attrName);
      if (!contextAttr) {
        newResultTypes.push_back(resultType);
        continue;
      }

      auto convertedType = convertType(resultType, contextAttr);
      if (convertedType == nullptr) return failure();
      newResultTypes.push_back(convertedType);
    }

    return success();
  }

 private:
  std::string attrName;
};

}  // namespace heir
}  // namespace mlir

namespace llvm {
// Enable hashing in dense map
template <>
struct DenseMapInfo<::mlir::heir::TypeAndAttribute> {
  static ::mlir::heir::TypeAndAttribute getEmptyKey() {
    return {DenseMapInfo<::mlir::Type>::getEmptyKey(),
            DenseMapInfo<::mlir::Attribute>::getEmptyKey()};
  }
  static ::mlir::heir::TypeAndAttribute getTombstoneKey() {
    return {DenseMapInfo<::mlir::Type>::getTombstoneKey(),
            DenseMapInfo<::mlir::Attribute>::getTombstoneKey()};
  }
  static unsigned getHashValue(const ::mlir::heir::TypeAndAttribute& val) {
    return llvm::hash_combine(val.type, val.attr);
  }
  static bool isEqual(const ::mlir::heir::TypeAndAttribute& lhs,
                      const ::mlir::heir::TypeAndAttribute& rhs) {
    return lhs.type == rhs.type && lhs.attr == rhs.attr;
  }
};

}  // namespace llvm

#endif  // LIB_UTILS_CONTEXTAWARETYPECONVERSION_H_
