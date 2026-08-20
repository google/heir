#include "lib/Dialect/TensorExt/Transforms/ImplementRotateAndReduce.h"

#include <memory>
#include <optional>
#include <string>

#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/IRMaterializingVisitor.h"
#include "lib/Kernel/KernelImplementation.h"
#include "lib/Kernel/Utils.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "llvm/include/llvm/Support/Debug.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"      // from @llvm-project
#include "mlir/include/mlir/IR/AsmState.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"          // from @llvm-project
#include "mlir/include/mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

#define DEBUG_TYPE "implement-rotate-and-reduce"

namespace mlir {
namespace heir {
namespace tensor_ext {

using ::mlir::heir::kernel::ArithmeticDagNode;
using ::mlir::heir::kernel::implementRotateAndReduce;
using ::mlir::heir::kernel::IRMaterializingVisitor;
using ::mlir::heir::kernel::SSAValue;

namespace {

// Gathers the given rows of a constant into a new constant of compactType.
static DenseElementsAttr gatherConstantRows(DenseElementsAttr attr,
                                            RankedTensorType compactType,
                                            ArrayRef<int64_t> indices) {
  if (attr.isSplat()) {
    return DenseElementsAttr::get(compactType, attr.getSplatValue<Attribute>());
  }
  auto attrType = cast<ShapedType>(attr.getType());
  int64_t numCols = attrType.getDimSize(1);
  // Sub-byte element types pack multiple elements per byte, so rows are not
  // byte-aligned; gather them element by element instead.
  if (attrType.getElementTypeBitWidth() < 8) {
    auto values = attr.getValues<Attribute>();
    SmallVector<Attribute> compact;
    compact.reserve(indices.size() * numCols);
    for (int64_t index : indices) {
      for (int64_t j = 0; j < numCols; ++j) {
        compact.push_back(values[index * numCols + j]);
      }
    }
    return DenseElementsAttr::get(compactType, compact);
  }
  ArrayRef<char> raw = attr.getRawData();
  int64_t rowBytes = raw.size() / attrType.getDimSize(0);
  SmallVector<char> compact;
  compact.reserve(indices.size() * rowBytes);
  for (int64_t index : indices) {
    llvm::append_range(compact, raw.slice(index * rowBytes, rowBytes));
  }
  return DenseElementsAttr::getFromRawBuffer(compactType, compact);
}

// Gathers the given rows of a resource-backed constant into a new resource of
// compactType. Returns nullptr when the element type is not byte-aligned, since
// a raw byte slice would then split an element.
static TypedAttr gatherResourceRows(DenseResourceElementsAttr attr,
                                    RankedTensorType compactType,
                                    ArrayRef<int64_t> indices) {
  Type elementType = compactType.getElementType();
  if (!elementType.isIntOrFloat() ||
      elementType.getIntOrFloatBitWidth() % 8 != 0) {
    return nullptr;
  }
  ArrayRef<char> raw = attr.getData();
  if (raw.empty()) return nullptr;

  int64_t numRows = cast<ShapedType>(attr.getType()).getDimSize(0);
  int64_t rowBytes = raw.size() / numRows;
  SmallVector<char> compact;
  compact.reserve(indices.size() * rowBytes);
  for (int64_t index : indices) {
    llvm::append_range(compact, raw.slice(index * rowBytes, rowBytes));
  }

  AsmResourceBlob* blob = attr.getRawHandle().getBlob();
  if (!blob) return nullptr;
  auto compactBlob = HeapAsmResourceBlob::allocateAndCopyWithAlign(
      ArrayRef<char>(compact.data(), compact.size()), blob->getDataAlignment(),
      /*dataIsMutable=*/false);
  std::string name = attr.getRawHandle().getKey().str() + "_gathered";
  return DenseResourceElementsAttr::get(compactType, name,
                                        std::move(compactBlob));
}

// Gathers the named rows of the packed matrix at compile time, or returns
// nullptr when the producer is not a constant these can read.
static TypedAttr gatherRowsIfConstant(Value diagonals,
                                      RankedTensorType compactType,
                                      ArrayRef<int64_t> indices) {
  DenseElementsAttr denseAttr;
  if (matchPattern(diagonals, m_Constant(&denseAttr))) {
    return gatherConstantRows(denseAttr, compactType, indices);
  }
  auto constantOp = diagonals.getDefiningOp<arith::ConstantOp>();
  if (!constantOp) return nullptr;
  if (auto resourceAttr =
          dyn_cast<DenseResourceElementsAttr>(constantOp.getValue())) {
    return gatherResourceRows(resourceAttr, compactType, indices);
  }
  return nullptr;
}

// A rotate_and_reduce marked as a linear transform maps directly onto
// kernel.linear_transform: the plaintexts are the generalized diagonals and
// tensor_ext.diagonal_indices records which ones are present (absent means
// row i is diagonal i).
LogicalResult convertToLinearTransform(RotateAndReduceOp op) {
  if (!op->hasAttr(TensorExtDialect::kLintransAttrName)) return failure();
  auto module = op->getParentOfType<ModuleOp>();
  if (!module) return failure();
  auto target = getTargetConfig(module);
  if (failed(target) || !target->has_kernel_linear_transform) {
    return failure();
  }
  if (!op.getPlaintexts()) return failure();

  OpBuilder builder(op);
  auto indicesAttr = op->getAttrOfType<DenseI32ArrayAttr>(
      TensorExtDialect::kDiagonalIndicesAttrName);
  SmallVector<int64_t> indices;
  if (indicesAttr) {
    llvm::append_range(indices, indicesAttr.asArrayRef());
  } else {
    for (int64_t i = 0, e = op.getSteps().getZExtValue(); i < e; ++i) {
      indices.push_back(i);
    }
  }

  // kernel.linear_transform's contract is positional: row k of the diagonals
  // operand is the diagonal named by diagonal_indices[k]. When the indices
  // name a subset of the packed matrix's rows, gather those rows out. For a
  // constant, dense or resource-backed, the gather folds into a compact
  // constant of the same form immediately, which is then all a later resource
  // externalization retains; otherwise it is an explicit cleartext gather.
  Value diagonals = op.getPlaintexts();
  auto diagonalsType = cast<RankedTensorType>(diagonals.getType());
  int64_t numRows = diagonalsType.getDimSize(0);
  DenseI64ArrayAttr sourceRowIndices;
  if (static_cast<int64_t>(indices.size()) < numRows) {
    auto compactType = RankedTensorType::get(
        {static_cast<int64_t>(indices.size()), diagonalsType.getDimSize(1)},
        diagonalsType.getElementType());
    if (TypedAttr gathered =
            gatherRowsIfConstant(diagonals, compactType, indices)) {
      diagonals = arith::ConstantOp::create(builder, op.getLoc(), gathered);
    } else {
      sourceRowIndices = builder.getDenseI64ArrayAttr(indices);
    }
  }

  auto linearTransformOp = kernel::LinearTransformOp::create(
      builder, op.getLoc(), op.getOutput().getType(), op.getTensor(), diagonals,
      builder.getDenseI64ArrayAttr(indices), sourceRowIndices,
      /*bsgs_ratio=*/nullptr);
  if (auto layout = op->getAttr(TensorExtDialect::kLayoutAttrName)) {
    linearTransformOp->setAttr(TensorExtDialect::kLayoutAttrName, layout);
  }
  op.getOutput().replaceAllUsesWith(linearTransformOp.getResult());
  op.erase();
  return success();
}

}  // namespace

#define GEN_PASS_DEF_IMPLEMENTROTATEANDREDUCE
#include "lib/Dialect/TensorExt/Transforms/Passes.h.inc"

LogicalResult convertRotateAndReduceOp(RotateAndReduceOp op, bool unroll) {
  LLVM_DEBUG(llvm::dbgs() << "Converting tensor_ext.rotate_and_reduce op: "
                          << op << "\n");
  TypedValue<RankedTensorType> input = op.getTensor();
  unsigned steps = op.getSteps().getZExtValue();
  unsigned period = op.getPeriod().getZExtValue();
  std::shared_ptr<ArithmeticDagNode<SSAValue>> implementedKernel;
  SSAValue vectorLeaf(input);
  std::optional<SSAValue> plaintextsLeaf = std::nullopt;

  if (op.getPlaintexts()) {
    plaintextsLeaf = std::optional<SSAValue>(op.getPlaintexts());
  }

  std::string reduceOp = "arith.addi";
  if (op.getReduceOp().has_value() && *op.getReduceOp() != nullptr) {
    reduceOp = op.getReduceOp()->getValue().str();
  }
  kernel::DagType dagType = kernel::mlirTypeToDagType(input.getType());
  implementedKernel = implementRotateAndReduce(
      vectorLeaf, plaintextsLeaf, period, steps, dagType, {}, reduceOp, unroll);
  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPointAfter(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  IRMaterializingVisitor visitor(input.getType());
  Value finalOutput = visitor.process(implementedKernel, b)[0];
  rewriter.replaceOp(op, finalOutput);
  return success();
}

struct ImplementRotateAndReduce
    : impl::ImplementRotateAndReduceBase<ImplementRotateAndReduce> {
  using ImplementRotateAndReduceBase::ImplementRotateAndReduceBase;

  void runOnOperation() override {
    getOperation()->walk([&](RotateAndReduceOp op) {
      // Hand the whole transform to the backend when it can evaluate one.
      if (succeeded(convertToLinearTransform(op))) return;
      if (failed(convertRotateAndReduceOp(op, unroll))) {
        op->emitOpError() << "failed to lower rotate_and_reduce op";
        signalPassFailure();
      }
    });
  }
};

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
