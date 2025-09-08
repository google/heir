#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_

include "lib/Dialect/TensorExt/IR/TensorExtDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/TensorEncoding.td"

class TensorExt_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TensorExt_Dialect, name, traits # [OpAsmAttrInterface]> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";

  string commonClassDeclaration = [{
   // OpAsmAttrInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # attrMnemonic # [{";
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];


  let extraClassDeclaration = commonClassDeclaration;
}

// TODO(#2047): replace tensor_ext.layout attribute with this (rename to remove New)
def TensorExt_NewLayoutAttr : TensorExt_Attr<"NewLayout", "new_layout"> {
  let summary = "The description of the layout of a data-semantic tensor.";
  let description = [{
    This attribute describes how a data-semantic tensor is laid out among a
    tensor of ciphertexts. The layout is described by an integer relation $(d,
    s)$, where $d$ is a multiset of data-semantic tensor indices and $s$ is a
    multiset of slot indices (or coefficient indices). The slot indices are
    defined by two indices: the ciphertext index and the slot index in that
    order. The elements of the relation are defined by a set of quasi-affine
    constraints.

    I.e., a point $((2, 3), (7, 0))$ in the relation says that the data entry
    at index $(2, 3)$ is placed in slot 0 of ciphertext 7. This could be
    defined as part of the relation by a constraint like `row + col + 2 - ct +
    slot = 0`.

    The attribute stores a string representation of the integer relation,
    which follows the ISL syntax for `isl_basic_map`. For example:

    ```mlir
    #vec_layout = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 1024 = 7 and i0 >= 0 and 0 >= i0 and slot >= 0 and 1023 >= slot and ct = 0 }">

    #mat_layout = #tensor_ext.new_layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 512 = 0 and (ct + slot - col) mod 512 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 1023 >= slot and 511 >= ct and 511 >= row and 511 >= col }">

    // Example with local (existential) variables.
    #layout = #tensor_ext.new_layout<"{[d0] -> [ct, slot] : exists d3, d4 : -ct + d3 = 0 and d0 - d4 * 1024 = 0 and -d0 + 31 >= 0 and d0 >= 0 and ct >= 0 and slot >= 0 and -slot + 1023 >= 0 and -d0 + d3 * 1024 + 1023 >= 0 and d0 - d3 * 1024 >= 0 and -d0 + d4 * 1024 + 1023 >= 0 and d0 - d4 * 1024 >= 0 }">
    ```
  }];
  let parameters = (ins "mlir::StringAttr":$layout);
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $layout `>`";

  let builders = [
    AttrBuilder<(ins "std::string":$layoutStr), [{
      return $_get($_ctxt, StringAttr::get($_ctxt, layoutStr));
    }]>,
  ];

  let extraClassDeclaration = commonClassDeclaration # [{
    presburger::IntegerRelation getIntegerRelation() const;

    std::string getLayoutStr() const {
      return getLayout().str();
    }

    static NewLayoutAttr getFromIntegerRelation(
      ::mlir::MLIRContext* context, const presburger::IntegerRelation &relation);
  }];
}

def TensorExt_OriginalTypeAttr : TensorExt_Attr<"OriginalType", "original_type"> {
  let summary = "The original type of a secret tensor whose layout has been converted to ciphertext semantics.";
  let description = [{
    This attribute is used to retain the original type of a secret tensor after
    its conversion to ciphertext semantics, i.e. after applying any padding or
    alignment to fill ciphertext data types. For example, if a
    `!secret.secret<tensor<32xi8>>` is laid out in a ciphertext with
    1024 slots, the new type would be `!secret.secret<tensor<1024xi8>>`
    with attribute `tensor_ext.original_type<!secret.secret<tensor<32xi8>>`.
  }];
  let parameters = (ins
    "::mlir::Type":$originalType,
    "::mlir::Attribute":$layout
  );
  let assemblyFormat =  "`<` struct(params) `>`";
}

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_
