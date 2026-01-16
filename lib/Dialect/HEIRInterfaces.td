#ifndef LIB_DIALECT_HEIR_INTERFACES_TD_
#define LIB_DIALECT_HEIR_INTERFACES_TD_

include "mlir/IR/OpBase.td"

def SecretTypeInterface : TypeInterface<"SecretTypeInterface"> {
  let cppNamespace = "::mlir::heir";
  let description = [{
    This interface marks that a type represents a secret value. This
    is an interface (beyond just the `secret.secret` type) so that
    concrete ciphertext types can also be marked as secret and reuse
    analyses that depend on `SecretnessAnalysis`.
  }];
}

def IncreasesMulDepthOpInterface : OpInterface<"IncreasesMulDepthOpInterface"> {
  let cppNamespace = "::mlir::heir";
  let description = [{
    A trait that signals whether an operation is a mul-like operation, used for
    generalizing multiplicative depth analysis.
  }];

  // No methods; We declare an interface instead of a Trait because we need to
  // attach the interface to upstream MLIR ops, and there is no way to do that
  // with Traits because Traits are intended to be statically checkable. Cf.
  // https://discourse.llvm.org/t/how-to-add-addtional-traits-to-existing-ops/62039
}

def LUTOpInterface : OpInterface<"LUTOpInterface"> {
  let cppNamespace = "::mlir::heir";
  let description = [{
    An interface that represents a general lookup table operation.
  }];

  let methods = [
    InterfaceMethod<
      "Gets lookup table inputs from most significant bit to least.",
      "std::optional<mlir::ValueRange>", "getLookupTableInputs"
    >,
  ];
}

def PlaintextOperandInterface : OpInterface<"PlaintextOperandInterface"> {
  let cppNamespace = "::mlir::heir";
  let description = [{
    An interface that identifies which operand indices of an operation may be plaintext.
    This is relevant primarily for a "plaintext-ciphertext" operation before lowering
    to scheme-specific APIs, such as an `arith.addi` or a `tensor_ext.rotate_and_reduce`.

    This interface is used in dataflow analyses to determine which operand of a
    ciphertext-plaintext op requires back-propagating level/dimension
    information to. Commutative ops for which multiple operands _may_ be plaintexts
    (e.g., `arith.addi`) should report that both operand indices may be plaintext,
    and the analysis is responsible for determining the secretness of the operands
    in question.

    Operands that are cleartext (i.e., never encoded) should not be included here,
    e.g., the `index` operand of a `tensor_ext.rotate`.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return a vector of indices describing which operands may be plaintexts.",
      /*retTy=*/"SmallVector<unsigned>",
      /*methodName=*/"maybePlaintextOperands",
      /*args=*/(ins ),
      /*defaultImplementation=*/[{ return {}; }]
    >,
  ];
}

def OperandLayoutRequirementOpInterface : OpInterface<"OperandLayoutRequirementOpInterface"> {
  let cppNamespace = "::mlir::heir";
  let description = [{
    An interface that signals which operands require a layout for
    implementation with an FHE kernel. This may also depend on whether
    the operand is secret.

    By default, all operands require a layout, even cleartext operands.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return true if this operand requires a layout",
      /*retTy=*/"bool",
      /*methodName=*/"operandRequiresLayout",
      /*args=*/(ins "unsigned":$operandIndex, "bool":$isSecret),
      /*defaultImplementation=*/[{ return true; }]
    >,
  ];
}

def ElementwiseByOperandOpInterface : OpInterface<"ElementwiseByOperandOpInterface"> {
  let cppNamespace = "::mlir::heir";
  let description = [{
    An interface that identifies the operation to be elementwise mappable,
    i.e., tensorizable and scalarizable, and specifies a subset of operands
    and results with respect to which this operation can be mapped.

    Non-tensor operands need not specify their mappability under this interface,
    and they are always replicated wholesale during elementwise mapping.

    Unlike `::mlir::Elementwise` and `::mlir::Scalarizable`, etc., this allows
    an operation to map across just some operands, while the other operands
    are kept wholesale. But like `::mlir::Elementwise`, this assumes that all
    tensors that are mapped over must have the same shape.

    An example of where this is used in HEIR is for scheme dialect operations
    like `relinearize`, which can be mapped across its first operand, but
    requires an additional key-switching key operand that is a tensor, but must
    be replicated and not mapped.

    This trait is designed to work with the `convert-elementwise-to-affine`
    pass.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return true if this operand can be mapped elementise",
      /*retTy=*/"bool",
      /*methodName=*/"operandIsMappable",
      /*args=*/(ins "unsigned":$operandIndex)
    >,

    // It may make sense some day to allow only a subset of op results to be
    // mapped, but for now we assume all results are tensors of the same shape
    // as the mapped operands.
  ];

  let verify = [{
    return ::mlir::heir::verifyElementwiseByOperandImpl(
      cast<ElementwiseByOperandOpInterface>(op));
  }];
}

def BatchVectorizableOpInterface : OpInterface<"BatchVectorizableOpInterface"> {
  let cppNamespace = "::mlir::heir";
  let description = [{
    An interface that identifies the operation to be batched, and specifies the
    conditions under which it can be batched with other operations and what
    batched operation it will be transformed into. This interface is used to
    support operations in the `batch-vectorize` pass.

    This interface is required to be on operations with the ElementwiseMappable
    or ElementwiseByOperandOpInterface. These interface determine which operands
    to map elementwise (or keep as wholesale scalars).
  }];

    let cppNamespace = "::mlir::heir";
  let description = [{
    An interface that signals which operands require a layout for
    implementation with an FHE kernel. This may also depend on whether
    the operand is secret.

    By default, all operands require a layout, even cleartext operands.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return true if the operation is compatible for batching with this",
      /*retTy=*/"bool",
      /*methodName=*/"isBatchCompatible",
      /*args=*/(ins "::mlir::Operation *":$rhs),
      /*body=*/[{}],
      /*defaultImplementation=*/[{
        ::mlir::Operation *lhs = $_op;
        return (lhs->getDialect() == rhs->getDialect() &&
            lhs->getResultTypes() == rhs->getResultTypes() &&
            lhs->getNumOperands() == rhs->getNumOperands());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the batched operation given a list of batchable operations",
      /*retTy=*/"FailureOr<::mlir::Operation *>",
      /*methodName=*/"buildBatchedOperation",
      /*args=*/(ins "::mlir::MLIRContext *":$context,
                    "::mlir::OpBuilder &":$builder,
                    "::mlir::SmallVector<::mlir::Value>":$vectorizedOperands,
                    "::mlir::SmallVector<::mlir::Operation*>":$operations)
    >,
  ];
}

def OperandAndResultAttrInterface : OpInterface<"OperandAndResultAttrInterface"> {
  let cppNamespace = "::mlir::heir";
  let description = [{
    This interfaces provides a convenience for attaching attributes to op
    arguments and results. This uses the existing attribute dictionary on the
    op, so that it can be attached to upstream ops without modifying their
    tablegen definition.
  }];
  let extraSharedClassDeclaration = [{
    // Use dictionary key names that are unlikely to cause conflicts.
    constexpr const static ::llvm::StringLiteral
        kOperandAttrsName = "__argattrs";
    constexpr const static ::llvm::StringLiteral
        kResultAttrsName = "__resattrs";

    // Returns an array attr where entry i contains the DictionaryAttr for operand i,
    // or nullptr if no operand attrs have been set
    ::mlir::ArrayAttr getAllOperandAttrsAttr() {
      return $_op->template getAttrOfType<::mlir::ArrayAttr>(kOperandAttrsName);
    }

    ::mlir::Attribute removeAllOperandAttrsAttr() {
      return $_op->removeAttr(kOperandAttrsName);
    }

    ::mlir::DictionaryAttr getOperandAttrDict(unsigned index) {
      assert(index < $_op->getNumOperands() && "invalid argument number");
      ::mlir::ArrayAttr attrs = getAllOperandAttrsAttr();
      ::mlir::DictionaryAttr argAttrs =
          attrs ? llvm::cast<DictionaryAttr>(attrs[index]) : ::mlir::DictionaryAttr();
      return argAttrs;
    }

    /// Return the specified attribute, if present, for the argument at 'index',
    /// null otherwise.
    ::mlir::Attribute getOperandAttr(unsigned index, ::mlir::StringAttr name) {
      auto argDict = getOperandAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    ::mlir::Attribute getOperandAttr(unsigned index, ::llvm::StringRef name) {
      auto argDict = getOperandAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getOperandAttrOfType(unsigned index, ::mlir::StringAttr name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getOperandAttr(index, name));
    }
    template <typename AttrClass>
    AttrClass getOperandAttrOfType(unsigned index, ::llvm::StringRef name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getOperandAttr(index, name));
    }

    void setOperandAttrsAttr(::mlir::ArrayAttr attr) {
      return $_op->setAttr(kOperandAttrsName, attr);
    }

    static bool isEmptyAttrDict(::mlir::Attribute attr) {
      return ::llvm::cast<::mlir::DictionaryAttr>(attr).empty();
    }

    void setOperandAttrDict(unsigned index, ::mlir::DictionaryAttr attr) {
      auto allAttrs = getAllOperandAttrsAttr();
      if (!allAttrs) {
        if (attr.empty())
          return;

        ::llvm::SmallVector<::mlir::Attribute, 4> newAttrs(
            $_op->getNumOperands(), ::mlir::DictionaryAttr::get($_op->getContext()));

        newAttrs[index] = attr;
        setOperandAttrsAttr(ArrayAttr::get($_op->getContext(), newAttrs));
        return;
      }

      if (allAttrs[index] == attr)
        return;

      // check if the new attr would be all empty dictionaries
      ::llvm::ArrayRef<::mlir::Attribute> rawAttrArray = allAttrs.getValue();
      if (attr.empty() &&
          ::llvm::all_of(rawAttrArray.take_front(index), isEmptyAttrDict) &&
          ::llvm::all_of(rawAttrArray.drop_front(index + 1), isEmptyAttrDict)) {
        removeAllOperandAttrsAttr();
        return;
      }

      // Otherwise, create a new attribute array with the updated dictionary.
      ::llvm::SmallVector<::mlir::Attribute, 8> newAttrs(rawAttrArray);
      newAttrs[index] = attr;
      setOperandAttrsAttr(ArrayAttr::get($_op->getContext(), newAttrs));
    }

    void setOperandAttr(unsigned index, ::mlir::StringAttr name, ::mlir::Attribute value) {
      ::mlir::NamedAttrList attributes(getOperandAttrDict(index));
      ::mlir::Attribute oldValue = attributes.set(name, value);
      // If the attribute changed, then set the new arg attribute list.
      if (value != oldValue)
        setOperandAttrDict(index, attributes.getDictionary(value.getContext()));
    }

    void setOperandAttr(unsigned index, ::llvm::StringRef name, ::mlir::Attribute value) {
      setOperandAttr(index, ::mlir::StringAttr::get(this->getOperation()->getContext(), name), value);
    }

    ::mlir::DictionaryAttr removeOperandAttrDict(unsigned index) {
      auto allAttrs = getAllOperandAttrsAttr();
      if (!allAttrs)
        return nullptr;

      auto result = cast<::mlir::DictionaryAttr>(allAttrs[index]);
      ::llvm::ArrayRef<::mlir::Attribute> rawAttrArray = allAttrs.getValue();
      if (::llvm::all_of(rawAttrArray.take_front(index), isEmptyAttrDict) &&
          ::llvm::all_of(rawAttrArray.drop_front(index + 1), isEmptyAttrDict)) {
        removeAllOperandAttrsAttr();
        return result;
      }
      ::llvm::SmallVector<::mlir::Attribute, 8> newAttrs(rawAttrArray);
      newAttrs[index] = ::mlir::DictionaryAttr::get($_op->getContext());
      setOperandAttrsAttr(ArrayAttr::get($_op->getContext(), newAttrs));
      return result;
    }

    /// Remove the attribute 'name' from the argument at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    ::mlir::Attribute removeOperandAttr(unsigned index, ::mlir::StringAttr name) {
      ::mlir::NamedAttrList attributes(getOperandAttrDict(index));
      ::mlir::Attribute removedAttr = attributes.erase(name);

      // If the attribute was removed, then update the argument dictionary.
      if (removedAttr)
        setOperandAttrDict(index, attributes.getDictionary(removedAttr.getContext()));
      return removedAttr;
    }
    ::mlir::Attribute removeOperandAttr(unsigned index, ::llvm::StringRef name) {
      return removeOperandAttr(
          index, ::mlir::StringAttr::get(this->getOperation()->getContext(), name));
    }

    //===------------------------------------------------------------------===//
    // Result Attributes
    //===------------------------------------------------------------------===//

    // Returns an array attr where entry i contains the DictionaryAttr for result i,
    // or nullptr if no result attrs have been set
    ::mlir::ArrayAttr getAllResultAttrsAttr() {
      return $_op->template getAttrOfType<::mlir::ArrayAttr>(kResultAttrsName);
    }

    ::mlir::Attribute removeAllResultAttrsAttr() {
      return $_op->removeAttr(kResultAttrsName);
    }

    ::mlir::DictionaryAttr getResultAttrDict(unsigned index) {
      assert(index < $_op->getNumResults() && "invalid argument number");
      ::mlir::ArrayAttr attrs = getAllResultAttrsAttr();
      ::mlir::DictionaryAttr argAttrs =
          attrs ? llvm::cast<DictionaryAttr>(attrs[index]) : ::mlir::DictionaryAttr();
      return argAttrs;
    }

    /// Return the specified attribute, if present, for the argument at 'index',
    /// null otherwise.
    ::mlir::Attribute getResultAttr(unsigned index, ::mlir::StringAttr name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    ::mlir::Attribute getResultAttr(unsigned index, ::llvm::StringRef name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, ::mlir::StringAttr name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getResultAttr(index, name));
    }
    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, ::llvm::StringRef name) {
      return ::llvm::dyn_cast_or_null<AttrClass>(getResultAttr(index, name));
    }

    void setResultAttrsAttr(::mlir::ArrayAttr attr) {
      return $_op->setAttr(kResultAttrsName, attr);
    }

    void setResultAttrDict(unsigned index, ::mlir::DictionaryAttr attr) {
      auto allAttrs = getAllResultAttrsAttr();
      if (!allAttrs) {
        if (attr.empty())
          return;

        ::llvm::SmallVector<::mlir::Attribute, 4> newAttrs(
            $_op->getNumResults(), ::mlir::DictionaryAttr::get($_op->getContext()));

        newAttrs[index] = attr;
        setResultAttrsAttr(ArrayAttr::get($_op->getContext(), newAttrs));
        return;
      }

      if (allAttrs[index] == attr)
        return;

      // check if the new attr would be all empty dictionaries
      ::llvm::ArrayRef<::mlir::Attribute> rawAttrArray = allAttrs.getValue();
      if (attr.empty() &&
          ::llvm::all_of(rawAttrArray.take_front(index), isEmptyAttrDict) &&
          ::llvm::all_of(rawAttrArray.drop_front(index + 1), isEmptyAttrDict)) {
        removeAllResultAttrsAttr();
        return;
      }

      // Otherwise, create a new attribute array with the updated dictionary.
      ::llvm::SmallVector<::mlir::Attribute, 8> newAttrs(rawAttrArray);
      newAttrs[index] = attr;
      setResultAttrsAttr(ArrayAttr::get($_op->getContext(), newAttrs));
    }

    void setResultAttr(unsigned index, ::mlir::StringAttr name, ::mlir::Attribute value) {
      ::mlir::NamedAttrList attributes(getResultAttrDict(index));
      ::mlir::Attribute oldValue = attributes.set(name, value);
      // If the attribute changed, then set the new arg attribute list.
      if (value != oldValue)
        setResultAttrDict(index, attributes.getDictionary(value.getContext()));
    }

    void setResultAttr(unsigned index, ::llvm::StringRef name, ::mlir::Attribute value) {
      setResultAttr(index,
                 ::mlir::StringAttr::get(this->getOperation()->getContext(), name),
                 value);
    }

    /// Remove the attribute 'name' from the argument at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    ::mlir::Attribute removeResultAttr(unsigned index, ::mlir::StringAttr name) {
      ::mlir::NamedAttrList attributes(getResultAttrDict(index));
      ::mlir::Attribute removedAttr = attributes.erase(name);

      // If the attribute was removed, then update the argument dictionary.
      if (removedAttr)
        setResultAttrDict(index, attributes.getDictionary(removedAttr.getContext()));
      return removedAttr;
    }
    ::mlir::Attribute removeResultAttr(unsigned index, ::llvm::StringRef name) {
      return removeResultAttr(
          index, ::mlir::StringAttr::get(this->getOperation()->getContext(), name));
    }
  }];
}

#endif  // LIB_DIALECT_HEIR_IR_HEIRINTERFACES_TD_
