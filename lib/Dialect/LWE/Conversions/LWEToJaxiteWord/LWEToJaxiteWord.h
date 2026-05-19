#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITEWORD_LWETOJAXITEWORD_H_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITEWORD_LWETOJAXITEWORD_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_DECL_LWETOJAXITEWORD
#include "lib/Dialect/LWE/Conversions/LWEToJaxiteWord/LWEToJaxiteWord.h.inc"

void registerLWEToJaxiteWordPasses();

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOJAXITEWORD_LWETOJAXITEWORD_H_
