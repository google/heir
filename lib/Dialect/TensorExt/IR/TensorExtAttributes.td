#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_

include "lib/Dialect/TensorExt/IR/TensorExtDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/TensorEncoding.td"

class TensorExt_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TensorExt_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

def SIMDPacking_Attr : TensorExt_Attr<"SIMDPacking", "simd_packing",
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
    tensor<1x16xi32, #packing>
    ```

    If the original tensor had values `[1, 2, 3, 4, 5, 6, 7]` then a tensor with
    this attribute contains the data `[1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6,
    7, 0]`.
  }];

  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$in,
    "::mlir::DenseI64ArrayAttr":$padding,
    "::mlir::DenseI64ArrayAttr":$out,
    DefaultValuedParameter<"int64_t", "0">:$padding_value
  );

  let assemblyFormat =  "`<` struct(params) `>`";
}

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_TD_
