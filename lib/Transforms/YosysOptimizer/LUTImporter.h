#ifndef HEIR_LIB_TRANSFORMS_YOSYSOPTIMIZER_LUTIMPORTER_H_
#define HEIR_LIB_TRANSFORMS_YOSYSOPTIMIZER_LUTIMPORTER_H_

#include "kernel/rtlil.h"  // from @at_clifford_yosys
#include "lib/Transforms/YosysOptimizer/RTLILImporter.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {

// LUTImporter implements the RTLILConfig for importing RTLIL that uses LUTs.
class LUTImporter : public RTLILImporter {
 public:
  LUTImporter(MLIRContext *context) : RTLILImporter(context) {}

 protected:
  Operation *createOp(Yosys::RTLIL::Cell *cell, SmallVector<Value, 4> &inputs,
                      ImplicitLocOpBuilder &b) const override;

  SmallVector<Yosys::RTLIL::SigSpec, 4> getInputs(
      Yosys::RTLIL::Cell *cell) const override;

  Yosys::RTLIL::SigSpec getOutput(Yosys::RTLIL::Cell *cell) const override;
};

}  // namespace heir
}  // namespace mlir

#endif  // HEIR_LIB_TRANSFORMS_YOSYSOPTIMIZER_LUTIMPORTER_H_
