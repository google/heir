#include "lib/Dialect/CGGI/IR/CGGIOps.h"

#include "mlir/include/mlir/IR/ValueRange.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cggi {

mlir::ValueRange Lut2Op::getLookupTableInputs() {
  return mlir::ValueRange{getB(), getA()};
}

mlir::ValueRange Lut3Op::getLookupTableInputs() {
  return mlir::ValueRange{getC(), getB(), getA()};
}

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
