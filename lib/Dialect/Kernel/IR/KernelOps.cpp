#include "lib/Dialect/Kernel/IR/KernelOps.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
// IWYU pragma: end_keep

// Generated definitions
#define GET_OP_CLASSES
#include "lib/Dialect/Kernel/IR/KernelOps.cpp.inc"

namespace {
uint32_t ceil_log2(uint32_t x) {
  if (x <= 1) return 0;
  return std::bit_width(x - 1);
}

// Ported from OpenFHE's ComputeDegreesPS in
// third_party/openfhe/src/pke/lib/scheme/ckksrns/ckksrns-utils.cpp
std::pair<uint32_t, uint32_t> computeDegreesPS(uint32_t n) {
  if (n == 0) return {0, 0};

  constexpr uint32_t UPPER_BOUND_PS = 2204;
  static const std::vector<std::pair<uint32_t, uint32_t>> rangemap = {
      {2, 1},    {11, 2},   {13, 3},   {17, 2},   {55, 3},  {59, 4},
      {76, 3},   {239, 4},  {247, 5},  {284, 4},  {991, 5}, {1007, 6},
      {1083, 5}, {2015, 6}, {2031, 7}, {2204, 6},
  };

  if (n <= UPPER_BOUND_PS) {
    uint32_t m = 0;
    for (const auto& entry : rangemap) {
      if (n <= entry.first) {
        m = entry.second;
        break;
      }
    }
    uint32_t k = n / ((1U << m) - 1) + 1;
    return {k, m};
  }

  // Heuristic for larger degrees
  std::vector<uint32_t> klist;
  std::vector<uint32_t> mlist;
  std::vector<uint32_t> multlist;
  for (uint32_t k = 1; k <= n; ++k) {
    double log2_n_k = std::log2(static_cast<double>(n) / k);
    uint32_t max_m = static_cast<uint32_t>(std::ceil(log2_n_k + 1) + 1);
    for (uint32_t m = 1; m <= max_m; ++m) {
      if (n < (k * ((1U << m) - 1))) {
        double log2_k = std::log2(k);
        double log2_sqrt = std::log2(std::sqrt(static_cast<double>(n) / 2));
        if (std::abs(std::floor(log2_k) - std::floor(log2_sqrt)) <= 1.0) {
          klist.push_back(k);
          mlist.push_back(m);
          multlist.push_back(k + 2 * m + (1U << (m - 1)) - 4);
        }
      }
    }
  }
  if (multlist.empty()) {
    return {1, static_cast<uint32_t>(std::ceil(std::log2(n + 1)))};
  }
  uint32_t minIndex =
      std::min_element(multlist.begin(), multlist.end()) - multlist.begin();
  return {klist[minIndex], mlist[minIndex]};
}
}  // namespace

namespace mlir {
namespace heir {
namespace kernel {

int EvalChebyshevOp::getLevelsToDrop() {
  BackendName backend = BackendName::Lattigo;

  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) {
    emitWarning(
        "eval_chebyshev op could not find enclosing module for backend "
        "determination; using lattigo by default");
  } else {
    FailureOr<CompilationTarget> targetConfig = getTargetConfig(module);
    if (failed(targetConfig)) {
      emitWarning(
          "eval_chebyshev op could not determine chosen backend; using lattigo "
          "by default");
    } else {
      backend = targetConfig->backendName;
    }
  }

  auto coefficients = getCoefficients().getValue();
  if (coefficients.empty()) {
    // The zero polynomial consumes no depth
    return 0;
  }

  int baseDepth = 0;
  uint32_t degree = coefficients.size() - 1;
  switch (backend) {
    case BackendName::Lattigo:
      baseDepth = std::bit_width(static_cast<uint64_t>(degree));
      break;
    case BackendName::OpenFHE:
      if (degree == 0) {
        baseDepth = 0;
      } else if (degree < 5) {
        baseDepth = ceil_log2(degree) + 1;
      } else {
        auto [k, m] = computeDegreesPS(degree);
        baseDepth = ceil_log2(k) + m;
      }
      break;
    default:
      baseDepth = 0;
  }

  return baseDepth;
}

::llvm::SmallVector<::mlir::OpOperand*> EvalChebyshevOp::getOperandsToReduce(
    const ::mlir::DataFlowSolver* solver) {
  return {&getOperation()->getOpOperand(0)};
}

int LinearTransformOp::getLevelsToDrop() { return 1; }

::llvm::SmallVector<::mlir::OpOperand*> LinearTransformOp::getOperandsToReduce(
    const ::mlir::DataFlowSolver* solver) {
  return {&getOperation()->getOpOperand(0)};
}

int ApplyLinearTransformOp::getLevelsToDrop() { return 1; }

::llvm::SmallVector<::mlir::OpOperand*>
ApplyLinearTransformOp::getOperandsToReduce(
    const ::mlir::DataFlowSolver* solver) {
  return {&getOperation()->getOpOperand(0)};
}

namespace {

// Returns the total slot capacity of a (possibly tensor-wrapped) ciphertext
// input, or nullopt when the type does not determine one.
std::optional<int64_t> getInputSlotSize(Type inputType) {
  auto inputRankedType = dyn_cast<RankedTensorType>(inputType);
  if (!inputRankedType) return std::nullopt;

  auto elementType = inputRankedType.getElementType();
  int64_t slotsPerCiphertext = 1;
  if (auto ctType = dyn_cast<lwe::LWECiphertextType>(elementType)) {
    auto plaintextSpace = ctType.getPlaintextSpace();
    auto ring = plaintextSpace.getRing();
    slotsPerCiphertext =
        ring.getPolynomialModulus().getPolynomial().getDegree();
    if (isa<lwe::InverseCanonicalEncodingAttr>(plaintextSpace.getEncoding())) {
      slotsPerCiphertext /= 2;
    }
  }

  if (inputRankedType.getRank() == 1) {
    return inputRankedType.getDimSize(0) * slotsPerCiphertext;
  }
  if (inputRankedType.getRank() == 2 && inputRankedType.getDimSize(0) == 1) {
    return inputRankedType.getDimSize(1) * slotsPerCiphertext;
  }
  return std::nullopt;
}

// Returns the modulus-chain level of a (possibly tensor-wrapped) LWE
// ciphertext type, or nullopt when the type carries no modulus chain.
std::optional<int64_t> getInputLevel(Type inputType) {
  auto ctType = dyn_cast<lwe::LWECiphertextType>(inputType);
  if (!ctType) {
    if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
      ctType = dyn_cast<lwe::LWECiphertextType>(tensorType.getElementType());
    }
  }
  if (!ctType) return std::nullopt;
  return lwe::getLevel(ctType);
}

LogicalResult verifyDiagonalRows(Operation* op, ShapedType diagonalsType,
                                 DenseI64ArrayAttr diagonalIndices,
                                 DenseI64ArrayAttr sourceRowIndices) {
  int64_t numDiagonals = diagonalsType.getDimSize(0);
  int64_t numIndices = diagonalIndices.size();
  if (!sourceRowIndices) {
    if (numDiagonals != numIndices) {
      return op->emitOpError("number of diagonals (")
             << numDiagonals << ") must match number of diagonal indices ("
             << numIndices << ")";
    }
    return success();
  }

  if (static_cast<int64_t>(sourceRowIndices.size()) != numIndices) {
    return op->emitOpError("number of source row indices (")
           << sourceRowIndices.size()
           << ") must match number of diagonal indices (" << numIndices << ")";
  }
  if (ShapedType::isDynamic(numDiagonals)) return success();
  for (int64_t row : sourceRowIndices.asArrayRef()) {
    if (row < 0 || row >= numDiagonals) {
      return op->emitOpError("source row index ")
             << row << " is out of bounds for " << numDiagonals
             << " diagonal rows";
    }
  }
  return success();
}

}  // namespace

LogicalResult PrepareLinearTransformOp::verify() {
  auto diagonalsType = dyn_cast<ShapedType>(getDiagonals().getType());
  if (!diagonalsType) {
    return emitOpError("diagonals must have a shaped type");
  }
  if (diagonalsType.getRank() != 2) {
    return emitOpError("diagonals must be a 2D tensor");
  }

  if (failed(verifyDiagonalRows(getOperation(), diagonalsType,
                                getDiagonalIndicesAttr(),
                                getSourceRowIndicesAttr())))
    return failure();

  int64_t slots = getPrepared().getType().getSlots();
  if (diagonalsType.getDimSize(1) > slots) {
    return emitOpError("diagonals slot size (")
           << diagonalsType.getDimSize(1)
           << ") exceeds the prepared slot count (" << slots << ")";
  }
  return success();
}

LogicalResult ApplyLinearTransformOp::verify() {
  PreparedLinearTransformType preparedType = getPrepared().getType();

  // A wrong level would silently evaluate a wrongly-scaled transform, so
  // require the prepared level to match the ciphertext exactly.
  std::optional<int64_t> inputLevel = getInputLevel(getInput().getType());
  if (inputLevel.has_value() && *inputLevel != preparedType.getLevel()) {
    return emitOpError("input ciphertext level (")
           << *inputLevel << ") does not match the prepared transform level ("
           << preparedType.getLevel() << ")";
  }

  std::optional<int64_t> inputSlots = getInputSlotSize(getInput().getType());
  if (inputSlots.has_value() && *inputSlots < preparedType.getSlots()) {
    return emitOpError("input slot size (")
           << *inputSlots << ") is smaller than the prepared slot count ("
           << preparedType.getSlots() << ")";
  }
  return success();
}

LogicalResult LinearTransformOp::verify() {
  auto inputType = getInput().getType();
  auto diagonalsType = dyn_cast<ShapedType>(getDiagonals().getType());
  if (!diagonalsType) {
    return emitOpError("diagonals must have a shaped type");
  }

  if (diagonalsType.getRank() != 2) {
    return emitOpError("diagonals must be a 2D tensor");
  }

  if (auto inputRankedType = dyn_cast<RankedTensorType>(inputType)) {
    int64_t inputSize = 0;
    auto elementType = inputRankedType.getElementType();
    int64_t slotsPerCiphertext = 1;
    if (auto ctType = dyn_cast<lwe::LWECiphertextType>(elementType)) {
      auto plaintextSpace = ctType.getPlaintextSpace();
      auto ring = plaintextSpace.getRing();
      slotsPerCiphertext =
          ring.getPolynomialModulus().getPolynomial().getDegree();
      if (isa<lwe::InverseCanonicalEncodingAttr>(
              plaintextSpace.getEncoding())) {
        slotsPerCiphertext /= 2;
      }
    }

    if (inputRankedType.getRank() == 1) {
      inputSize = inputRankedType.getDimSize(0) * slotsPerCiphertext;
    } else if (inputRankedType.getRank() == 2) {
      if (inputRankedType.getDimSize(0) != 1) {
        return emitOpError(
            "input tensor batch dimension (first dimension) must be 1");
      }
      inputSize = inputRankedType.getDimSize(1) * slotsPerCiphertext;
    } else {
      return emitOpError("input must be 1D or 2D ranked tensor");
    }

    // The diagonals may be narrower than the ciphertext: the transform then
    // acts on the leading slots and the backend's encoder zero-fills the rest.
    int64_t diagonalSlotSize = diagonalsType.getDimSize(1);
    if (inputSize < diagonalSlotSize) {
      return emitOpError("input slot size (")
             << inputSize << ") is smaller than diagonals slot size ("
             << diagonalSlotSize << ")";
    }
  }

  if (failed(verifyDiagonalRows(getOperation(), diagonalsType,
                                getDiagonalIndicesAttr(),
                                getSourceRowIndicesAttr())))
    return failure();

  return success();
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
