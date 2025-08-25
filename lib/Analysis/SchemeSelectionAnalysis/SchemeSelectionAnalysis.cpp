#include "lib/Analysis/SchemeSelectionAnalysis/SchemeSelectionAnalysis.h"

#include <algorithm>
#include <cassert>
#include <functional>

#include "SchemeSelectionAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"                // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "SchemeSelection"

namespace mlir {
namespace heir {

LogicalResult SchemeSelectionAnalysis::visitOperation(
    Operation *op, ArrayRef<const SchemeInfoLattice *> operands,
    ArrayRef<SchemeInfoLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting: " << op->getName() << ". \n");

  return success();
}

int getCountAttribute(Operation *op, StringRef attrName) {
  Attribute attr = op->getAttr(attrName);
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr)) {
    return intAttr.getInt();
  }
  return -1;
}

int calcArithMetric(NatureOfComputation noc) {
  auto arithBoolOpFactor = 1;
  auto arithBitOpFactor = 1;
  auto arithIntOpFactor = 1;
  auto arithRealOpFactor = 2;
  auto arithCmpOpFactor = 10;
  auto arithNonlinOpFactor = 10;

  auto result = arithBoolOpFactor * noc.getBoolOpsCount() +
                arithBitOpFactor * noc.getBitOpsCount() +
                arithIntOpFactor * noc.getIntArithOpsCount() +
                arithRealOpFactor * noc.getRealArithOpsCount() +
                arithCmpOpFactor * noc.getCmpOpsCount() +
                arithNonlinOpFactor * noc.getNonLinOpsCount();

  return result;
}

int calcBitMetric(NatureOfComputation noc) {
  auto bitBoolOpFactor = 1;
  auto bitBitOpFactor = 1;
  auto bitIntOpFactor = 3;
  auto bitRealOpFactor = 6;
  auto bitCmpOpFactor = 4;
  auto bitNonlinOpFactor = 6;

  auto result = bitBoolOpFactor * noc.getBoolOpsCount() +
                bitBitOpFactor * noc.getBitOpsCount() +
                bitIntOpFactor * noc.getIntArithOpsCount() +
                bitRealOpFactor * noc.getRealArithOpsCount() +
                bitCmpOpFactor * noc.getCmpOpsCount() +
                bitNonlinOpFactor * noc.getNonLinOpsCount();
  return result;
}

NatureOfComputation getNatureOfComputation(Operation *op) {
  auto numBoolOps = getCountAttribute(op, numBoolOpsAttrName);
  auto numBitOps = getCountAttribute(op, numBitOpsAttrName);
  auto numIntArithOps = getCountAttribute(op, numIntArithOpsAttrName);
  auto numRealArithOps = getCountAttribute(op, numRealArithOpsAttrName);
  auto numCmpOps = getCountAttribute(op, numCmpOpsAttrName);
  auto numNonlinOps = getCountAttribute(op, numNonLinOpsAttrName);
  return NatureOfComputation(numBoolOps, numBitOps, numIntArithOps,
                             numRealArithOps, numCmpOps, numNonlinOps);
}

std::string annotateModuleWithScheme(Operation *top, DataFlowSolver *solver) {
  llvm::DenseMap<StringRef, unsigned> freq;

  LLVM_DEBUG(llvm::dbgs() << "Top operation is: " << top->getName() << "\n");
  top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    for (auto attr : funcOp->getAttrs()) {
      LLVM_DEBUG(llvm::dbgs() << "Func attr: " << attr.getName()
                              << "with value " << attr.getValue() << "\n");
    }

    auto count = getNatureOfComputation(funcOp);
    LLVM_DEBUG(llvm::dbgs() << "Nature of Computation: " << count << "\n");
    auto arithWeight = calcArithMetric(count);
    auto bitWeight = calcBitMetric(count);
    LLVM_DEBUG(llvm::dbgs() << "Arith weight is: " << arithWeight
                            << " and bit weight is: " << bitWeight << "\n");
    StringRef scheme;
    if (arithWeight < bitWeight) {
      if (count.getIntArithOpsCount() >= count.getRealArithOpsCount()) {
        LLVM_DEBUG(llvm::dbgs() << "Scheme selected for fun: "
                                << funcOp->getName() << " is BGV/BFV\n");
        scheme = BGV;
      } else {
        // scheme CKKS
        LLVM_DEBUG(llvm::dbgs() << "Scheme selected for fun: "
                                << funcOp->getName() << " is CKKS\n");
        scheme = CKKS;
      }
    } else {
      // scheme CGGI
      LLVM_DEBUG(llvm::dbgs() << "Scheme selected for fun: "
                              << funcOp->getName() << " is CGGI\n");
      scheme = CGGI;
    }
    ++freq[scheme];
  });

  StringRef selection;
  unsigned bestCount = 0;
  for (auto &kv : freq)
    if (kv.second > bestCount) {
      selection = kv.getFirst();
      bestCount = kv.second;
    }
  LLVM_DEBUG(llvm::dbgs() << "Best scheme is " << selection << "\n");

  return selection.str();
}
}  // namespace heir
}  // namespace mlir
