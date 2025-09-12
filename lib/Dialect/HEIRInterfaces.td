include "mlir/IR/OpBase.td"

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

def OperandLayoutRequirementOpInterface : OpInterface<"OperandLayoutRequirementOpInterface"> {
  let cppNamespace = "::mlir::heir";
  let description = [{
    An interface that signals which operands require a layout for
    implementation with an FHE kernel.

    By default, all operands require a layout, even cleartext operands.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return true if this operand requires a layout",
      /*retTy=*/"bool",
      /*methodName=*/"operandRequiresLayout",
      /*args=*/(ins "unsigned":$operandIndex),
      /*defaultImplementation=*/[{ return true; }]
    >,
  ];
}

def LayoutConversionHoistableOpInterface : OpInterface<"LayoutConversionHoistableOpInterface"> {
  let cppNamespace = "::mlir::heir";
  let description = [{
    An interface that abstracts the common information required for the
    `layout-optimization` pass to evaluate the feasibility and cost of hoisting
    a `convert_layout` op through a given operation.
  }];

  let methods = [
    InterfaceMethod<
      "Get hoisters for the op.",
      "std::vector<::mlir::heir::Hoister>",
      "getHoisters",
      (ins "::mlir::heir::tensor_ext::ConvertLayoutOp":$convertLayoutOp)
    >,
  ];

  // TODO(#1888): figure out how to get OpInterface verifier to run
  // automatically.
  // let verify = [{
  //   ::mlir::heir::KernelName kernelName = ::mlir::heir::KernelName::Trivial;
  //   auto attrName = ::mlir::heir::secret::SecretDialect::kKernelAttrName;
  //   if ($_op->hasAttr(attrName)) {
  //     auto attr = $_op->getAttrOfType<::mlir::heir::secret::KernelAttr>(attrName);
  //     kernelName = attr.getName();
  //   }
  //   if (!::mlir::heir::isSupportedKernel($_op, kernelName)) {
  //     return $_op->emitOpError()
  //            << "has unsupported kernel ''";
  //   }
  //   return success();
  // }];
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
