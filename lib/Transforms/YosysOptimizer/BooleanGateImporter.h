#ifndef HEIR_LIB_TRANSFORMS_YOSYSOPTIMIZER_BOOLEANGATEIMPORTER_H_
#define HEIR_LIB_TRANSFORMS_YOSYSOPTIMIZER_BOOLEANGATEIMPORTER_H_

#include "lib/Transforms/YosysOptimizer/RTLILImporter.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

// Block clang-format from reordering
// clang-format off
#include "kernel/rtlil.h" // from @at_clifford_yosys
// clang-format on
namespace mlir {
namespace heir {

// BooleanGateImporter implements the RTLILConfig for importing RTLIL that uses
// boolean gates.
class BooleanGateImporter : public RTLILImporter {
 public:
  BooleanGateImporter(MLIRContext *context) : RTLILImporter(context) {}

  ~BooleanGateImporter() override = default;

 protected:
  Operation *createOp(Yosys::RTLIL::Cell *cell, SmallVector<Value> &inputs,
                      ImplicitLocOpBuilder &b) const override;

  SmallVector<Yosys::RTLIL::SigSpec> getInputs(
      Yosys::RTLIL::Cell *cell) const override;

  Yosys::RTLIL::SigSpec getOutput(Yosys::RTLIL::Cell *cell) const override;
};

}  // namespace heir
}  // namespace mlir

#endif  // HEIR_LIB_TRANSFORMS_YOSYSOPTIMIZER_BOOLEANGATEIMPORTER_H_
