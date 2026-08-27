#ifndef LIB_CONVERSIONS_MATHTOEMITC_MATHTOEMITC_H_
#define LIB_CONVERSIONS_MATHTOEMITC_MATHTOEMITC_H_

#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace mlir::heir {

void registerConvertMathToEmitCInterface(DialectRegistry& registry);

}  // namespace mlir::heir

#endif  // LIB_CONVERSIONS_MATHTOEMITC_MATHTOEMITC_H_
