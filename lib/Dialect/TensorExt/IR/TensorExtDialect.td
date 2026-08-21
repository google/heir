#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def TensorExt_Dialect : Dialect {
  let name = "tensor_ext";
  let description = [{
    The `tensor_ext` dialect contains operations on plaintext tensors that
    correspond to the computation model of certain FHE schemes, but are
    unlikely to be upstreamed to MLIR due to their specificity to FHE.
  }];

  let cppNamespace = "::mlir::heir::tensor_ext";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    constexpr const static ::llvm::StringLiteral
        kLayoutAttrName = "tensor_ext.layout";
    constexpr const static ::llvm::StringLiteral
        kOriginalTypeAttrName = "tensor_ext.original_type";
    // Marks a rotate_and_reduce that is a linear transform the backend can take
    // directly, so implement-rotate-and-reduce leaves it alone.
    constexpr const static ::llvm::StringLiteral
        kLintransAttrName = "tensor_ext.lintrans";
    // On such an op, when the plaintexts hold only the nonzero diagonals of a
    // mostly-zero matrix, records each row's original diagonal offset
    // (DenseI32ArrayAttr). Absent means row i is diagonal i.
    constexpr const static ::llvm::StringLiteral
        kDiagonalIndicesAttrName = "tensor_ext.diagonal_indices";
  }];


  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_
