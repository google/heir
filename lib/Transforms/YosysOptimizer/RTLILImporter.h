#ifndef LIB_TRANSFORMS_YOSYSOPTIMIZER_RTLILIMPORTER_H_
#define LIB_TRANSFORMS_YOSYSOPTIMIZER_RTLILIMPORTER_H_

#include <optional>
#include <sstream>
#include <string>

#include "llvm/include/llvm/ADT/MapVector.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/StringMap.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

// Block clang-format from reordering
// clang-format off
#include "kernel/rtlil.h" // from @at_clifford_yosys
// clang-format on

namespace mlir {
namespace heir {

// Returns a list of cell names that are topologically ordered using the Yosys
// toder output. This is extracted from the lines containing cells in the
// output:
// -- Running command `torder -stop * P*;' --

// 14. Executing TORDER pass (print cells in topological order).
// module test_add
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$168
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$170
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$169
//   cell $abc$167$auto$blifparse.cc:525:parse_blif$171
llvm::SmallVector<std::string, 10> getTopologicalOrder(
    std::stringstream &torderOutput);

class RTLILImporter {
 public:
  RTLILImporter(MLIRContext *context) : context(context) {}

  // importModule imports an RTLIL module to an MLIR function using the provided
  // config. cellOrdering provides topologically sorted list of cells that can
  // be used to sequentially create the MLIR representation. A resultType is
  // also passed to specify the output shape. For example, if a flattened 8 bit
  // vector is returned from the module, it may be assembled into a memref<8xi1>
  // or memref<2x4xi1>.
  func::FuncOp importModule(Yosys::RTLIL::Module *module,
                            const SmallVector<std::string, 10> &cellOrdering,
                            std::optional<SmallVector<Type>> resultTypes);

  virtual ~RTLILImporter() = default;

 protected:
  // cellToOp converts an RTLIL cell to an MLIR operation.
  virtual Operation *createOp(Yosys::RTLIL::Cell *cell,
                              SmallVector<Value> &inputs,
                              ImplicitLocOpBuilder &b) const = 0;

  // Returns a list of RTLIL cell inputs.
  virtual SmallVector<Yosys::RTLIL::SigSpec> getInputs(
      Yosys::RTLIL::Cell *cell) const = 0;

  // Returns an RTLIL cell output.
  virtual Yosys::RTLIL::SigSpec getOutput(Yosys::RTLIL::Cell *cell) const = 0;

 private:
  MLIRContext *context;

  llvm::StringMap<Value> wireNameToValue;
  Value getWireValue(Yosys::RTLIL::Wire *wire);
  void addWireValue(Yosys::RTLIL::Wire *wire, Value value);

  // getBit gets the MLIR Value corresponding to the given connection. This
  // assumes that the connection is a single bit.
  Value getBit(
      const Yosys::RTLIL::SigSpec &conn, ImplicitLocOpBuilder &b,
      llvm::MapVector<Yosys::RTLIL::Wire *, SmallVector<Value>> &retBitValues);

  // addResultBit assigns an mlir result to the result connection.
  void addResultBit(
      const Yosys::RTLIL::SigSpec &conn, Value result,
      llvm::MapVector<Yosys::RTLIL::Wire *, SmallVector<Value>> &retBitValues);
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_YOSYSOPTIMIZER_RTLILIMPORTER_H_
