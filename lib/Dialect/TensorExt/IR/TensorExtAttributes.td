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

  let extraClassDeclaration = [{
    // OpAsmAttrInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # attrMnemonic # [{";
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

// TODO(#1592): remove and remove align-tensor-sizes
def TensorExt_SIMDPackingAttr : TensorExt_Attr<"SIMDPacking", "simd_packing",
      [DeclareAttrInterfaceMethods<VerifiableTensorEncoding>]> {
  let summary = "An attribute describing the SIMD packing of a tensor.";
  let description = [{
    This attribute is used as the encoding attribute on a tensor. It describes
    the transformations that were applied to an input tensor to pack it into the
    given tensor.

    The `in` attribute describes the shape of the original tensor.
    The following transformations are applied to the input tensor.

      1. Padding is applied first. The `padding` attribute is an array with the
         same size as the input tensor shape. Padding is applied at the end of
         the array using the `padding_value` attribute (default zero). The
         result after zero padding should be a power of two.

      2. The padded result is replicated or split to fill the output tensor
         shape.

    For example,

    ```
    #packing = #tensor_ext.simd_packing<
      in = [7],
      padding = [1],
      padding_value = 0,
      out = [16],
    >
    ```

    may be used on a tensor type like

    ```
    tensor<16xi32>
    ```

    This indicates that the original input tensor was a `tensor<7xi32>`, and if
    it had values `[1, 2, 3, 4, 5, 6, 7]` then the packed tensor contains the
    data `[1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0]`.
  }];

  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$in,
    "::mlir::DenseI64ArrayAttr":$padding,
    "::mlir::DenseI64ArrayAttr":$out,
    DefaultValuedParameter<"int64_t", "0">:$padding_value
  );

  let assemblyFormat =  "`<` struct(params) `>`";
}

def TensorExt_AlignmentAttr : TensorExt_Attr<"Alignment", "alignment"> {
  let summary = "An attribute describing padding and alignment of a tensor.";
  let description = [{
    This attribute is used to describe how a data-semantic value is padded
    and replicated to align its size before applying a ciphertext layout
    (see `tensor_ext.layout`).

    It describes transformations to be applied to an input tensor or scalar.

    The `in` attribute describes the shape of the original value. An empty list
    for `in` indicates a scalar value that must be materialized as a tensor.
    The following transformations are applied to the input tensor.

    1. New unit dimensions are inserted to match the number of dimensions
       of the output tensor. The `insertedDims` attribute is an array
       specifying which axes in the output tensor are new unit dimensions. By
       default no new dimensions are inserted.
    2. Padding is applied. The `padding` attribute is an array with the same
       size as the output tensor shape, which each entry denoting the number of
       values to pad at the end of that axis. The `paddingValue` attribute
       describes the value used for padding, and must be set if `padding` is set.
    3. The padded result is replicated to fill the output tensor shape.


    Example:

    ```mlir
    #repl = #tensor_ext.alignment<
      in = [7],
      padding = [1],
      paddingValue = 0:i32,
      out = [16],
    >
    ```

    This indicates that the original input tensor is a `tensor<7xi32>`, and it
    is being padded and replicated into a `tensor<16xi32>`. If the input tensor
    has values `[1, 2, 3, 4, 5, 6, 7]` then the packed tensor contains the data
    `[1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0]`.

    Example:

    ```mlir
    #repl = #tensor_ext.alignment<
      in = [3, 10],
      padding = [1, 6],
      paddingValue = 0:i32,
      out = [32, 32],
    >
    ```

    This indicates that the original input tensor is a `tensor<3x10xi32>`, and
    it is being padded and replicated into a `tensor<32x32xi32>`. One row of
    zeros and six columns of zeros are added to the input tensor, and then it
    is repeated once along the column dimension and four times along the row
    dimension to fill the 32x32 output tensor.

    Example:

    ```mlir
    #repl = #tensor_ext.alignment<
      in = [10],
      insertedDims = [0],
      padding = [0, 6],
      paddingValue = 0:i32,
      out = [32, 32],
    >
    ```

    This indicates that the original input tensor is a `tensor<10xi32>`, and it
    is being padded and replicated into a `tensor<32x32xi32>`. First a unit
    dimension is inserted as the first axis, then zeros are padded along the
    second axis to make a `tensor<1x16xi32>`. Finally, it is replicated twice
    along the columns and 32 times along the rows to fill the 32x32 output
    tensor.

    Example:

    ```mlir
    #repl = #tensor_ext.alignment<
      in = [],
      insertedDims = [0],
      out = [1],
    >
    ```

    This indicates that the original input is a scalar such as `i32`, and it is
    elevated to a `tensor<1xi32>` before being replicated to fill the
    ciphertext shape.
  }];

  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$in,
    "::mlir::DenseI64ArrayAttr":$out,
    DefaultValuedParameter<"::mlir::DenseI64ArrayAttr", "$_builder.getDenseI64ArrayAttr({})">:$insertedDims,
    DefaultValuedParameter<"::mlir::DenseI64ArrayAttr", "$_builder.getDenseI64ArrayAttr({})">:$padding,
    DefaultValuedParameter<"TypedAttr", "nullptr">:$paddingValue
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::DenseI64ArrayAttr":$in,
                                        "::mlir::DenseI64ArrayAttr":$out), [{
      return $_get(
        in.getContext(),
        in,
        out,
        DenseI64ArrayAttr::get(in.getContext(), {}),
        DenseI64ArrayAttr::get(in.getContext(), {}),
        nullptr);
    }]>,

    AttrBuilder<(ins "::llvm::ArrayRef<int64_t>":$in,
                     "::llvm::ArrayRef<int64_t>":$out), [{
      return $_get(
        $_ctxt,
        DenseI64ArrayAttr::get($_ctxt, in),
        DenseI64ArrayAttr::get($_ctxt, out),
        DenseI64ArrayAttr::get($_ctxt, {}),
        DenseI64ArrayAttr::get($_ctxt, {}),
        nullptr
      );
    }]>,
  ];

  let assemblyFormat =  "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TensorExt_LayoutAttr : TensorExt_Attr<"Layout", "layout"> {
  let summary = "The description of the layout of a data-semantic tensor.";
  let description = [{
    This attribute describes how a data-semantic tensor is laid out
    among a tensor of ciphertexts. The layout is described by an affine map
    mapping the data-semantic tensor indices to the ciphertext tensor indices,
    where the trailing affine map's result expressions mark the slot index
    (or indices) of the ciphertext.

    The layout may include optional padding and alignment of the data-semantic
    tensor before the layout is applied. This would be required if, for
    example, a `tensor<4xi32>` is to be laid out in a `tensor<16xi32>`
    ciphertext tensor. The data-semantic tensor must be extended somehow to
    fill the 16 slots. If an alignment attribute is not provided, then
    lowerings may raise errors if there is no unambiguous way to align the
    tensor.
  }];
  let parameters = (ins
    "::mlir::AffineMap":$map,
    OptionalParameter<"::mlir::heir::tensor_ext::AlignmentAttr">:$alignment
  );
  let assemblyFormat =  "`<` struct(params) `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::AffineMap":$map,
                                        "::mlir::heir::tensor_ext::AlignmentAttr":$alignment), [{
      return $_get(map.getContext(), map, alignment);
    }]>,
    AttrBuilderWithInferredContext<(ins "::mlir::AffineMap":$map), [{
      return $_get(map.getContext(), map, nullptr);
    }]>,
  ];
  let genVerifyDecl = 1;
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
    TensorExt_LayoutAttr:$layout
  );
  let assemblyFormat =  "`<` struct(params) `>`";
}

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_
