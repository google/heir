#include "lib/Dialect/CGGI/IR/CGGIOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cggi {

std::optional<ValueRange> Lut2Op::getLookupTableInputs() {
  return ValueRange{getB(), getA()};
}

std::optional<ValueRange> Lut3Op::getLookupTableInputs() {
  return ValueRange{getC(), getB(), getA()};
}

std::optional<ValueRange> LutLinCombOp::getLookupTableInputs() {
  return ValueRange{getInputs()};
}

std::optional<ValueRange> PackedLut3Op::getLookupTableInputs() {
  return std::nullopt;
}

LogicalResult LutLinCombOp::verify() {
  if (getInputs().size() != getCoefficients().size())
    return emitOpError("number of coefficients must match number of inputs");

  lwe::NewLWECiphertextType type = dyn_cast<lwe::NewLWECiphertextType>(
      getElementTypeOrSelf(getOutput().getType()));
  // Tablegen allows AnyType due to error using Variadic on TypeOrContainer
  // types.
  if (!type) return emitOpError("expected LWE ciphertext element type");
  auto plaintextBitwidth = type.getPlaintextSpace()
                               .getRing()
                               .getCoefficientType()
                               .getIntOrFloatBitWidth();

  int64_t maxCoeff = (1 << plaintextBitwidth) - 1;
  for (auto c : getCoefficients()) {
    if (c > maxCoeff) {
      InFlightDiagnostic diag =
          emitOpError("coefficient pushes error bits into message space");
      diag.attachNote() << "coefficient is " << c;
      diag.attachNote() << "largest allowable coefficient is " << maxCoeff;
      return diag;
    }
  }

  if (getLookupTable().getValue().getActiveBits() > maxCoeff + 1) {
    InFlightDiagnostic diag =
        emitOpError("LUT is larger than available cleartext bit width");
    diag.attachNote() << "LUT has "
                      << getLookupTable().getValue().getActiveBits()
                      << " active bits";
    diag.attachNote() << "max LUT size is " << maxCoeff + 1 << " bits";
    return diag;
  }

  return success();
}

LogicalResult ProgrammableBootstrapOp::verify() {
  lwe::NewLWECiphertextType type = cast<lwe::NewLWECiphertextType>(
      getElementTypeOrSelf(getOutput().getType()));
  auto plaintextBitwidth = type.getPlaintextSpace()
                               .getRing()
                               .getCoefficientType()
                               .getIntOrFloatBitWidth();

  int64_t maxCoeff = (1 << plaintextBitwidth) - 1;
  if (getLookupTable().getValue().getActiveBits() > maxCoeff + 1) {
    InFlightDiagnostic diag =
        emitOpError("LUT is larger than available cleartext bit width");
    diag.attachNote() << "LUT has "
                      << getLookupTable().getValue().getActiveBits()
                      << " active bits";
    diag.attachNote() << "max LUT size is " << maxCoeff + 1 << " bits";
    return diag;
  }

  return success();
}

LogicalResult MultiLutLinCombOp::verify() {
  if (getInputs().size() != getCoefficients().size())
    return emitOpError("number of coefficients must match number of inputs");
  if (getOutputs().size() != getLookupTables().size())
    return emitOpError("number of outputs must match number of LUTs");

  lwe::NewLWECiphertextType type =
      cast<lwe::NewLWECiphertextType>(getOutputs().front().getType());
  auto plaintextBitwidth = type.getPlaintextSpace()
                               .getRing()
                               .getCoefficientType()
                               .getIntOrFloatBitWidth();

  int64_t maxCoeff = (1 << plaintextBitwidth) - 1;
  for (auto c : getCoefficients()) {
    if (c > maxCoeff) {
      InFlightDiagnostic diag =
          emitOpError("coefficient pushes error bits into message space");
      diag.attachNote() << "coefficient is " << c;
      diag.attachNote() << "largest allowable coefficient is " << maxCoeff;
      return diag;
    }
  }

  for (int64_t lut : getLookupTables()) {
    APInt apintLut = APInt(64, lut);
    if (apintLut.getActiveBits() > maxCoeff + 1) {
      InFlightDiagnostic diag =
          emitOpError("LUT is larger than available cleartext bit width");
      diag.attachNote() << "LUT has " << apintLut.getActiveBits()
                        << " active bits";
      diag.attachNote() << "max LUT size is " << maxCoeff + 1 << " bits";
      return diag;
    }
  }

  return success();
}

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
