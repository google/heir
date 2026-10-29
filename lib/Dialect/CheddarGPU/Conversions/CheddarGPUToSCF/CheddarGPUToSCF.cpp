#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToSCF/CheddarGPUToSCF.h"

#include <cstdint>
#include <utility>

#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToSCF/HoistPatterns.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

#define GEN_PASS_DEF_CHEDDARGPUTOSCF
#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToSCF/CheddarGPUToSCF.h.inc"

namespace {

// Helper: computes reverse of x over bits (using table/shifts)
uint32_t bitReverse(uint32_t x, uint32_t bits) {
  uint32_t rev = 0;
  for (uint32_t b = 0; b < bits; ++b) {
    if ((x >> b) & 1) {
      rev |= (1u << (bits - 1 - b));
    }
  }
  return rev;
}

// Computes Cheddar automorphism src index:
// x_rev = bitReverse(x, log_n + 1) + 1
// src_rev = bitReverse(x_rev * galois - 1, log_n + 1)
uint32_t getAutomorphismSrcIndex(uint32_t x, uint32_t logN, uint64_t galois) {
  uint32_t bits = logN + 1;
  uint32_t xRev = bitReverse(x, bits) + 1;
  uint64_t modMask = (1ULL << bits) - 1;
  uint64_t prod = (static_cast<uint64_t>(xRev) * galois - 1) & modMask;
  return bitReverse(static_cast<uint32_t>(prod), bits);
}

// Gets active limb offset and count for an op with an optional limb_range
// attribute.
std::pair<int64_t, int64_t> getLimbRange(std::optional<ArrayRef<int64_t>> range,
                                         int64_t totalLimbs) {
  if (range && range->size() == 2) {
    return {(*range)[0], (*range)[1]};
  }
  return {0, totalLimbs};
}

// Structure holding module parameters
struct ModuleParams {
  ParamsAttr params;
  uint32_t word;
  uint32_t logN;
  int64_t degree;
};

std::optional<ModuleParams> getModuleParams(Operation* op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module) return std::nullopt;
  auto params = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
  if (!params) return std::nullopt;
  return ModuleParams{params, params.getWord(), params.getLogN(),
                      1LL << params.getLogN()};
}

// Base pattern providing helper methods for emitting per-limb coefficient loops
template <typename OpTy>
struct CheddarGPUOpConversion : public OpRewritePattern<OpTy> {
  CheddarGPUOpConversion(MLIRContext* context)
      : OpRewritePattern<OpTy>(context) {}

  void emitPerLimbCoeffLoop(
      OpTy op, PatternRewriter& rewriter, const ModuleParams& modParams,
      int64_t limbStart, int64_t limbCount,
      llvm::function_ref<void(ImplicitLocOpBuilder& b, int64_t limbIdx,
                              uint64_t modulus, mod_arith::ModArithType modType,
                              Value coeffIdx)>
          bodyFn) const {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Type intType = IntegerType::get(b.getContext(), modParams.word);

    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value cDegree = arith::ConstantIndexOp::create(b, modParams.degree);
    Value c1 = arith::ConstantIndexOp::create(b, 1);

    int64_t totalLimbs = 0;
    if (op->getNumResults() > 0 &&
        isa<MemRefType>(op->getResult(0).getType())) {
      totalLimbs = cast<MemRefType>(op->getResult(0).getType()).getDimSize(0);
    } else if (op->getNumOperands() > 0 &&
               isa<MemRefType>(op->getOperand(0).getType())) {
      totalLimbs = cast<MemRefType>(op->getOperand(0).getType()).getDimSize(0);
    }

    int64_t numAux = 0;
    if (auto numAuxAttr = op->template getAttrOfType<IntegerAttr>("num_aux")) {
      numAux = numAuxAttr.getInt();
    }
    int64_t primeOffset = 0;
    if (auto primeOffsetAttr =
            op->template getAttrOfType<IntegerAttr>("prime_offset")) {
      primeOffset = primeOffsetAttr.getInt();
    }
    int64_t numQ = totalLimbs - numAux;
    int64_t totalPrimes = modParams.params.getNumLimbs();
    int64_t alpha = modParams.params.getAux().size();
    int64_t auxStart = totalPrimes - alpha;

    for (int64_t l = limbStart; l < limbStart + limbCount; ++l) {
      bool isAux = (numAux > 0 && l >= numQ);
      int64_t absLimb = isAux ? ((l - numQ) + auxStart) : (l + primeOffset);
      uint64_t q = modParams.params.getPrimeAt(absLimb);
      auto modAttr = IntegerAttr::get(intType, APInt(modParams.word, q));
      auto modType = mod_arith::ModArithType::get(b.getContext(), modAttr);

      scf::ForOp::create(b, c0, cDegree, c1, ValueRange{},
                         [&](OpBuilder& nestedBuilder, Location loc,
                             Value coeffIdx, ValueRange /*iterArgs*/) {
                           ImplicitLocOpBuilder nb(loc, nestedBuilder);
                           bodyFn(nb, l, q, modType, coeffIdx);
                           scf::YieldOp::create(nb);
                         });
    }
  }
};

// 0. CopyOp
struct ConvertCopyOp : public CheddarGPUOpConversion<CopyOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto inType = cast<MemRefType>(op.getIn().getType());
    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));
    auto [inStart, inCount] =
        getLimbRange(op.getInLimbRange(), inType.getDimSize(0));

    int64_t count = std::min(outCount, inCount);
    int64_t degree = outType.getDimSize(1);

    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value cDegree = arith::ConstantIndexOp::create(b, degree);

    for (int64_t l = 0; l < count; ++l) {
      Value inL = arith::ConstantIndexOp::create(b, inStart + l);
      Value outL = arith::ConstantIndexOp::create(b, outStart + l);
      scf::ForOp::create(b, c0, cDegree, c1, ValueRange{},
                         [&](OpBuilder& nestedBuilder, Location nestedLoc,
                             Value xIdx, ValueRange) {
                           ImplicitLocOpBuilder nb(nestedLoc, nestedBuilder);
                           Value val = memref::LoadOp::create(
                               nb, op.getIn(), ValueRange{inL, xIdx});
                           memref::StoreOp::create(nb, val, op.getOut(),
                                                   ValueRange{outL, xIdx});
                           scf::YieldOp::create(nb);
                         });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// 0b. ZeroPadOp: Zero-fill limb range [start, end)
struct ConvertZeroPadOp : public CheddarGPUOpConversion<ZeroPadOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(ZeroPadOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;

    auto range = op.getLimbRange();
    int64_t start = range[0];
    int64_t end = range[1];
    int64_t count = end - start;
    if (count <= 0) {
      rewriter.eraseOp(op);
      return success();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value cDegree = arith::ConstantIndexOp::create(b, degree);
    Value zeroVal = arith::ConstantIntOp::create(b, 0, word);

    for (int64_t l = start; l < end; ++l) {
      Value outL = arith::ConstantIndexOp::create(b, l);
      scf::ForOp::create(b, c0, cDegree, c1, ValueRange{},
                         [&](OpBuilder& nestedBuilder, Location nestedLoc,
                             Value xIdx, ValueRange) {
                           ImplicitLocOpBuilder nb(nestedLoc, nestedBuilder);
                           memref::StoreOp::create(nb, zeroVal, op.getOut(),
                                                   ValueRange{outL, xIdx});
                           scf::YieldOp::create(nb);
                         });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// 1. AddOp
struct ConvertAddOp : public CheddarGPUOpConversion<AddOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value lhsVal =
              memref::LoadOp::create(b, op.getLhs(), ValueRange{lVal, i});
          Value rhsVal =
              memref::LoadOp::create(b, op.getRhs(), ValueRange{lVal, i});

          Value lhsMod = mod_arith::EncapsulateOp::create(b, modType, lhsVal);
          Value rhsMod = mod_arith::EncapsulateOp::create(b, modType, rhsVal);
          Value sumMod = mod_arith::AddOp::create(b, modType, lhsMod, rhsMod);
          Value res = mod_arith::LiftOp::create(b, lhsVal.getType(), sumMod,
                                                mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, res, op.getOut(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 2. SubOp
struct ConvertSubOp : public CheddarGPUOpConversion<SubOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value lhsVal =
              memref::LoadOp::create(b, op.getLhs(), ValueRange{lVal, i});
          Value rhsVal =
              memref::LoadOp::create(b, op.getRhs(), ValueRange{lVal, i});

          Value lhsMod = mod_arith::EncapsulateOp::create(b, modType, lhsVal);
          Value rhsMod = mod_arith::EncapsulateOp::create(b, modType, rhsVal);
          Value diffMod = mod_arith::SubOp::create(b, modType, lhsMod, rhsMod);
          Value res = mod_arith::LiftOp::create(b, lhsVal.getType(), diffMod,
                                                mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, res, op.getOut(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 3. NegOp
struct ConvertNegOp : public CheddarGPUOpConversion<NegOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value inVal =
              memref::LoadOp::create(b, op.getIn(), ValueRange{lVal, i});
          Value zeroVal = arith::ConstantOp::create(
              b, inVal.getType(), b.getIntegerAttr(inVal.getType(), 0));

          Value zeroMod = mod_arith::EncapsulateOp::create(b, modType, zeroVal);
          Value inMod = mod_arith::EncapsulateOp::create(b, modType, inVal);
          Value negMod = mod_arith::SubOp::create(b, modType, zeroMod, inMod);
          Value res = mod_arith::LiftOp::create(b, inVal.getType(), negMod,
                                                mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, res, op.getOut(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 3b. Add2Op
struct ConvertAdd2Op : public CheddarGPUOpConversion<Add2Op> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(Add2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value lhsBxVal =
              memref::LoadOp::create(b, op.getLhsBx(), ValueRange{lVal, i});
          Value rhsBxVal =
              memref::LoadOp::create(b, op.getRhsBx(), ValueRange{lVal, i});
          Value lhsAxVal =
              memref::LoadOp::create(b, op.getLhsAx(), ValueRange{lVal, i});
          Value rhsAxVal =
              memref::LoadOp::create(b, op.getRhsAx(), ValueRange{lVal, i});

          Value lhsBxMod =
              mod_arith::EncapsulateOp::create(b, modType, lhsBxVal);
          Value rhsBxMod =
              mod_arith::EncapsulateOp::create(b, modType, rhsBxVal);
          Value sumBxMod =
              mod_arith::AddOp::create(b, modType, lhsBxMod, rhsBxMod);
          Value resBx = mod_arith::LiftOp::create(
              b, lhsBxVal.getType(), sumBxMod, mod_arith::LiftType::STANDARD);

          Value lhsAxMod =
              mod_arith::EncapsulateOp::create(b, modType, lhsAxVal);
          Value rhsAxMod =
              mod_arith::EncapsulateOp::create(b, modType, rhsAxVal);
          Value sumAxMod =
              mod_arith::AddOp::create(b, modType, lhsAxMod, rhsAxMod);
          Value resAx = mod_arith::LiftOp::create(
              b, lhsAxVal.getType(), sumAxMod, mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, resBx, op.getOutBx(), ValueRange{lVal, i});
          memref::StoreOp::create(b, resAx, op.getOutAx(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 3c. Sub2Op
struct ConvertSub2Op : public CheddarGPUOpConversion<Sub2Op> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(Sub2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value lhsBxVal =
              memref::LoadOp::create(b, op.getLhsBx(), ValueRange{lVal, i});
          Value rhsBxVal =
              memref::LoadOp::create(b, op.getRhsBx(), ValueRange{lVal, i});
          Value lhsAxVal =
              memref::LoadOp::create(b, op.getLhsAx(), ValueRange{lVal, i});
          Value rhsAxVal =
              memref::LoadOp::create(b, op.getRhsAx(), ValueRange{lVal, i});

          Value lhsBxMod =
              mod_arith::EncapsulateOp::create(b, modType, lhsBxVal);
          Value rhsBxMod =
              mod_arith::EncapsulateOp::create(b, modType, rhsBxVal);
          Value diffBxMod =
              mod_arith::SubOp::create(b, modType, lhsBxMod, rhsBxMod);
          Value resBx = mod_arith::LiftOp::create(
              b, lhsBxVal.getType(), diffBxMod, mod_arith::LiftType::STANDARD);

          Value lhsAxMod =
              mod_arith::EncapsulateOp::create(b, modType, lhsAxVal);
          Value rhsAxMod =
              mod_arith::EncapsulateOp::create(b, modType, rhsAxVal);
          Value diffAxMod =
              mod_arith::SubOp::create(b, modType, lhsAxMod, rhsAxMod);
          Value resAx = mod_arith::LiftOp::create(
              b, lhsAxVal.getType(), diffAxMod, mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, resBx, op.getOutBx(), ValueRange{lVal, i});
          memref::StoreOp::create(b, resAx, op.getOutAx(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 3d. Neg2Op
struct ConvertNeg2Op : public CheddarGPUOpConversion<Neg2Op> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(Neg2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value inBxVal =
              memref::LoadOp::create(b, op.getInBx(), ValueRange{lVal, i});
          Value inAxVal =
              memref::LoadOp::create(b, op.getInAx(), ValueRange{lVal, i});
          Value zeroVal = arith::ConstantOp::create(
              b, inBxVal.getType(), b.getIntegerAttr(inBxVal.getType(), 0));

          Value zeroMod = mod_arith::EncapsulateOp::create(b, modType, zeroVal);
          Value inBxMod = mod_arith::EncapsulateOp::create(b, modType, inBxVal);
          Value negBxMod =
              mod_arith::SubOp::create(b, modType, zeroMod, inBxMod);
          Value resBx = mod_arith::LiftOp::create(
              b, inBxVal.getType(), negBxMod, mod_arith::LiftType::STANDARD);

          Value inAxMod = mod_arith::EncapsulateOp::create(b, modType, inAxVal);
          Value negAxMod =
              mod_arith::SubOp::create(b, modType, zeroMod, inAxMod);
          Value resAx = mod_arith::LiftOp::create(
              b, inAxVal.getType(), negAxMod, mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, resBx, op.getOutBx(), ValueRange{lVal, i});
          memref::StoreOp::create(b, resAx, op.getOutAx(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 4. AddConstOp
struct ConvertAddConstOp : public CheddarGPUOpConversion<AddConstOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(AddConstOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value inVal =
              memref::LoadOp::create(b, op.getIn(), ValueRange{lVal, i});
          Value cstVal =
              memref::LoadOp::create(b, op.getCst(), ValueRange{lVal});

          Value inMod = mod_arith::EncapsulateOp::create(b, modType, inVal);
          Value cstMod = mod_arith::EncapsulateOp::create(b, modType, cstVal);
          Value sumMod = mod_arith::AddOp::create(b, modType, inMod, cstMod);
          Value res = mod_arith::LiftOp::create(b, inVal.getType(), sumMod,
                                                mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, res, op.getOut(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 5. SubConstOp
struct ConvertSubConstOp : public CheddarGPUOpConversion<SubConstOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(SubConstOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value inVal =
              memref::LoadOp::create(b, op.getIn(), ValueRange{lVal, i});
          Value cstVal =
              memref::LoadOp::create(b, op.getCst(), ValueRange{lVal});

          Value inMod = mod_arith::EncapsulateOp::create(b, modType, inVal);
          Value cstMod = mod_arith::EncapsulateOp::create(b, modType, cstVal);
          Value diffMod = mod_arith::SubOp::create(b, modType, inMod, cstMod);
          Value res = mod_arith::LiftOp::create(b, inVal.getType(), diffMod,
                                                mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, res, op.getOut(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 6. SubOppositeConstOp
struct ConvertSubOppositeConstOp
    : public CheddarGPUOpConversion<SubOppositeConstOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(SubOppositeConstOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value inVal =
              memref::LoadOp::create(b, op.getIn(), ValueRange{lVal, i});
          Value cstVal =
              memref::LoadOp::create(b, op.getCst(), ValueRange{lVal});

          Value inMod = mod_arith::EncapsulateOp::create(b, modType, inVal);
          Value cstMod = mod_arith::EncapsulateOp::create(b, modType, cstVal);
          Value diffMod = mod_arith::SubOp::create(b, modType, cstMod, inMod);
          Value res = mod_arith::LiftOp::create(b, inVal.getType(), diffMod,
                                                mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, res, op.getOut(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 7. MulOp
struct ConvertMulOp : public CheddarGPUOpConversion<MulOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value lhsVal =
              memref::LoadOp::create(b, op.getLhs(), ValueRange{lVal, i});
          Value rhsVal =
              memref::LoadOp::create(b, op.getRhs(), ValueRange{lVal, i});

          Value lhsMod = mod_arith::EncapsulateOp::create(b, modType, lhsVal);
          Value rhsMod = mod_arith::EncapsulateOp::create(b, modType, rhsVal);
          Value prodMod =
              mod_arith::MontMulOp::create(b, modType, lhsMod, rhsMod);
          Value res = mod_arith::LiftOp::create(b, lhsVal.getType(), prodMod,
                                                mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, res, op.getOut(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 8. MulConstOp
struct ConvertMulConstOp : public CheddarGPUOpConversion<MulConstOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(MulConstOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value inVal =
              memref::LoadOp::create(b, op.getIn(), ValueRange{lVal, i});
          Value cstVal =
              memref::LoadOp::create(b, op.getCst(), ValueRange{lVal});

          Value inMod = mod_arith::EncapsulateOp::create(b, modType, inVal);
          Value cstMod = mod_arith::EncapsulateOp::create(b, modType, cstVal);
          Value prodMod =
              mod_arith::MontMulOp::create(b, modType, inMod, cstMod);
          Value res = mod_arith::LiftOp::create(b, inVal.getType(), prodMod,
                                                mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, res, op.getOut(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 9. CPAccumOp
struct ConvertCPAccumOp : public CheddarGPUOpConversion<CPAccumOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(CPAccumOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    auto [lStart, lCount] = getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t primeOffset = op.getPrimeOffset().value_or(0);
    int64_t numQ = lCount - numAux;

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Type elemType = outType.getElementType();

          Value accMod = nullptr;
          for (size_t k = 0; k < op.getCtSrcs().size(); ++k) {
            Value ct = op.getCtSrcs()[k];
            Value common = op.getCommonSrcs()[k];
            auto commonType = cast<MemRefType>(common.getType());
            int64_t ctTotalLimbs = cast<MemRefType>(ct.getType()).getDimSize(0);

            Value ctLVal;
            if (ctTotalLimbs > outTotalLimbs) {
              int64_t localLimb = l - lStart;
              bool isAux = (numAux > 0 && localLimb >= numQ);
              int64_t ctNumQ = ctTotalLimbs - numAux;
              int64_t ctLimbIdx = isAux ? (ctNumQ + (localLimb - numQ))
                                        : (localLimb + primeOffset);
              ctLVal = arith::ConstantIndexOp::create(b, ctLimbIdx);
            } else {
              ctLVal = lVal;
            }
            Value ctVal = memref::LoadOp::create(b, ct, ValueRange{ctLVal, i});
            Value commonVal;
            if (commonType.getRank() == 1) {
              commonVal = memref::LoadOp::create(b, common, ValueRange{lVal});
            } else {
              commonVal =
                  memref::LoadOp::create(b, common, ValueRange{lVal, i});
            }

            Value ctMod = mod_arith::EncapsulateOp::create(b, modType, ctVal);
            Value commonMod =
                mod_arith::EncapsulateOp::create(b, modType, commonVal);
            Value prodMod =
                mod_arith::MontMulOp::create(b, modType, ctMod, commonMod);

            if (!accMod) {
              accMod = prodMod;
            } else {
              accMod = mod_arith::AddOp::create(b, modType, accMod, prodMod);
            }
          }

          Value res = mod_arith::LiftOp::create(b, elemType, accMod,
                                                mod_arith::LiftType::STANDARD);
          memref::StoreOp::create(b, res, op.getOut(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 10. CPAccumAddOp
struct ConvertCPAccumAddOp : public CheddarGPUOpConversion<CPAccumAddOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(CPAccumAddOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    auto [lStart, lCount] = getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t primeOffset = op.getPrimeOffset().value_or(0);
    int64_t numQ = lCount - numAux;

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Type elemType = outType.getElementType();

          Value baseVal =
              memref::LoadOp::create(b, op.getAcc(), ValueRange{lVal, i});
          Value accMod = mod_arith::EncapsulateOp::create(b, modType, baseVal);

          for (size_t k = 0; k < op.getCtSrcs().size(); ++k) {
            Value ct = op.getCtSrcs()[k];
            Value common = op.getCommonSrcs()[k];
            auto commonType = cast<MemRefType>(common.getType());
            int64_t ctTotalLimbs = cast<MemRefType>(ct.getType()).getDimSize(0);

            Value ctLVal;
            if (ctTotalLimbs > outTotalLimbs) {
              int64_t localLimb = l - lStart;
              bool isAux = (numAux > 0 && localLimb >= numQ);
              int64_t ctNumQ = ctTotalLimbs - numAux;
              int64_t ctLimbIdx = isAux ? (ctNumQ + (localLimb - numQ))
                                        : (localLimb + primeOffset);
              ctLVal = arith::ConstantIndexOp::create(b, ctLimbIdx);
            } else {
              ctLVal = lVal;
            }
            Value ctVal = memref::LoadOp::create(b, ct, ValueRange{ctLVal, i});
            Value commonVal;
            if (commonType.getRank() == 1) {
              commonVal = memref::LoadOp::create(b, common, ValueRange{lVal});
            } else {
              commonVal =
                  memref::LoadOp::create(b, common, ValueRange{lVal, i});
            }

            Value ctMod = mod_arith::EncapsulateOp::create(b, modType, ctVal);
            Value commonMod =
                mod_arith::EncapsulateOp::create(b, modType, commonVal);
            Value prodMod =
                mod_arith::MontMulOp::create(b, modType, ctMod, commonMod);
            accMod = mod_arith::AddOp::create(b, modType, accMod, prodMod);
          }

          Value res = mod_arith::LiftOp::create(b, elemType, accMod,
                                                mod_arith::LiftType::STANDARD);
          memref::StoreOp::create(b, res, op.getOut(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 10b. CPAccum2Op
struct ConvertCPAccum2Op : public CheddarGPUOpConversion<CPAccum2Op> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(CPAccum2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    auto [lStart, lCount] = getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t primeOffset = op.getPrimeOffset().value_or(0);
    int64_t numQ = lCount - numAux;

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Type elemType = outType.getElementType();

          Value accBxMod = nullptr;
          Value accAxMod = nullptr;
          for (size_t k = 0; k < op.getCommonSrcs().size(); ++k) {
            Value ctBx = op.getCtBxSrcs()[k];
            Value ctAx = op.getCtAxSrcs()[k];
            Value common = op.getCommonSrcs()[k];
            auto commonType = cast<MemRefType>(common.getType());
            int64_t ctTotalLimbs =
                cast<MemRefType>(ctBx.getType()).getDimSize(0);

            Value ctLVal;
            if (ctTotalLimbs > outTotalLimbs) {
              int64_t localLimb = l - lStart;
              bool isAux = (numAux > 0 && localLimb >= numQ);
              int64_t ctNumQ = ctTotalLimbs - numAux;
              int64_t ctLimbIdx = isAux ? (ctNumQ + (localLimb - numQ))
                                        : (localLimb + primeOffset);
              ctLVal = arith::ConstantIndexOp::create(b, ctLimbIdx);
            } else {
              ctLVal = lVal;
            }
            Value ctBxVal =
                memref::LoadOp::create(b, ctBx, ValueRange{ctLVal, i});
            Value ctAxVal =
                memref::LoadOp::create(b, ctAx, ValueRange{ctLVal, i});
            Value commonVal;
            if (commonType.getRank() == 1) {
              commonVal = memref::LoadOp::create(b, common, ValueRange{lVal});
            } else {
              commonVal =
                  memref::LoadOp::create(b, common, ValueRange{lVal, i});
            }

            Value ctBxMod =
                mod_arith::EncapsulateOp::create(b, modType, ctBxVal);
            Value ctAxMod =
                mod_arith::EncapsulateOp::create(b, modType, ctAxVal);
            Value commonMod =
                mod_arith::EncapsulateOp::create(b, modType, commonVal);

            Value prodBxMod =
                mod_arith::MontMulOp::create(b, modType, ctBxMod, commonMod);
            Value prodAxMod =
                mod_arith::MontMulOp::create(b, modType, ctAxMod, commonMod);

            if (!accBxMod) {
              accBxMod = prodBxMod;
              accAxMod = prodAxMod;
            } else {
              accBxMod =
                  mod_arith::AddOp::create(b, modType, accBxMod, prodBxMod);
              accAxMod =
                  mod_arith::AddOp::create(b, modType, accAxMod, prodAxMod);
            }
          }

          Value resBx = mod_arith::LiftOp::create(
              b, elemType, accBxMod, mod_arith::LiftType::STANDARD);
          Value resAx = mod_arith::LiftOp::create(
              b, elemType, accAxMod, mod_arith::LiftType::STANDARD);
          memref::StoreOp::create(b, resBx, op.getOutBx(), ValueRange{lVal, i});
          memref::StoreOp::create(b, resAx, op.getOutAx(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 10c. CPAccumAdd2Op
struct ConvertCPAccumAdd2Op : public CheddarGPUOpConversion<CPAccumAdd2Op> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(CPAccumAdd2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    auto [lStart, lCount] = getLimbRange(op.getOutLimbRange(), outTotalLimbs);

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t primeOffset = op.getPrimeOffset().value_or(0);
    int64_t numQ = lCount - numAux;

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Type elemType = outType.getElementType();

          Value baseBxVal =
              memref::LoadOp::create(b, op.getAccBx(), ValueRange{lVal, i});
          Value baseAxVal =
              memref::LoadOp::create(b, op.getAccAx(), ValueRange{lVal, i});
          Value accBxMod =
              mod_arith::EncapsulateOp::create(b, modType, baseBxVal);
          Value accAxMod =
              mod_arith::EncapsulateOp::create(b, modType, baseAxVal);

          for (size_t k = 0; k < op.getCommonSrcs().size(); ++k) {
            Value ctBx = op.getCtBxSrcs()[k];
            Value ctAx = op.getCtAxSrcs()[k];
            Value common = op.getCommonSrcs()[k];
            auto commonType = cast<MemRefType>(common.getType());
            int64_t ctTotalLimbs =
                cast<MemRefType>(ctBx.getType()).getDimSize(0);

            Value ctLVal;
            if (ctTotalLimbs > outTotalLimbs) {
              int64_t localLimb = l - lStart;
              bool isAux = (numAux > 0 && localLimb >= numQ);
              int64_t ctNumQ = ctTotalLimbs - numAux;
              int64_t ctLimbIdx = isAux ? (ctNumQ + (localLimb - numQ))
                                        : (localLimb + primeOffset);
              ctLVal = arith::ConstantIndexOp::create(b, ctLimbIdx);
            } else {
              ctLVal = lVal;
            }
            Value ctBxVal =
                memref::LoadOp::create(b, ctBx, ValueRange{ctLVal, i});
            Value ctAxVal =
                memref::LoadOp::create(b, ctAx, ValueRange{ctLVal, i});
            Value commonVal;
            if (commonType.getRank() == 1) {
              commonVal = memref::LoadOp::create(b, common, ValueRange{lVal});
            } else {
              commonVal =
                  memref::LoadOp::create(b, common, ValueRange{lVal, i});
            }

            Value ctBxMod =
                mod_arith::EncapsulateOp::create(b, modType, ctBxVal);
            Value ctAxMod =
                mod_arith::EncapsulateOp::create(b, modType, ctAxVal);
            Value commonMod =
                mod_arith::EncapsulateOp::create(b, modType, commonVal);

            Value prodBxMod =
                mod_arith::MontMulOp::create(b, modType, ctBxMod, commonMod);
            Value prodAxMod =
                mod_arith::MontMulOp::create(b, modType, ctAxMod, commonMod);

            accBxMod =
                mod_arith::AddOp::create(b, modType, accBxMod, prodBxMod);
            accAxMod =
                mod_arith::AddOp::create(b, modType, accAxMod, prodAxMod);
          }

          Value resBx = mod_arith::LiftOp::create(
              b, elemType, accBxMod, mod_arith::LiftType::STANDARD);
          Value resAx = mod_arith::LiftOp::create(
              b, elemType, accAxMod, mod_arith::LiftType::STANDARD);
          memref::StoreOp::create(b, resBx, op.getOutBx(), ValueRange{lVal, i});
          memref::StoreOp::create(b, resAx, op.getOutAx(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 10d. ModUpToMaxOp: Broadcast and reduce polynomial limb across primes
struct ConvertModUpToMaxOp : public CheddarGPUOpConversion<ModUpToMaxOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(ModUpToMaxOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t outTotalLimbs = outType.getDimSize(0);
    int64_t degree = modParams->degree;
    uint32_t word = modParams->word;
    size_t totalLimbs = modParams->params.getNumLimbs();

    auto [outStart, outCount] =
        getLimbRange(op.getOutLimbRange(), outTotalLimbs);
    if (outCount == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t alpha = modParams->params.getAux().size();
    int64_t numQ = outCount - numAux;
    int64_t auxStart = totalLimbs - alpha;
    int64_t primeOffset = op.getPrimeOffset().value_or(0);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value cDegree = arith::ConstantIndexOp::create(b, degree);

    for (int64_t l = 0; l < outCount; ++l) {
      bool isAux = (numAux > 0 && l >= numQ);
      int64_t absLimb = isAux ? ((l - numQ) + auxStart) : (l + primeOffset);
      uint64_t q = modParams->params.getPrimeAt(absLimb);

      Value outL = arith::ConstantIndexOp::create(b, outStart + l);
      Value qVal = arith::ConstantIntOp::create(b, q, word);

      scf::ForOp::create(b, c0, cDegree, c1, ValueRange{},
                         [&](OpBuilder& nestedBuilder, Location nestedLoc,
                             Value xIdx, ValueRange) {
                           ImplicitLocOpBuilder nb(nestedLoc, nestedBuilder);
                           Value inVal = memref::LoadOp::create(
                               nb, op.getIn(), ValueRange{c0, xIdx});
                           Value res = arith::RemUIOp::create(nb, inVal, qVal);
                           memref::StoreOp::create(nb, res, op.getOut(),
                                                   ValueRange{outL, xIdx});
                           scf::YieldOp::create(nb);
                         });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// 11. TensorOp: Karatsuba degree-1 ciphertext product
struct ConvertTensorOp : public CheddarGPUOpConversion<TensorOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(TensorOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Type elemType = outType.getElementType();

          Value b1Val =
              memref::LoadOp::create(b, op.getSrc1Bx(), ValueRange{lVal, i});
          Value a1Val =
              memref::LoadOp::create(b, op.getSrc1Ax(), ValueRange{lVal, i});
          Value b2Val =
              memref::LoadOp::create(b, op.getSrc2Bx(), ValueRange{lVal, i});
          Value a2Val =
              memref::LoadOp::create(b, op.getSrc2Ax(), ValueRange{lVal, i});

          Value b1Mod = mod_arith::EncapsulateOp::create(b, modType, b1Val);
          Value a1Mod = mod_arith::EncapsulateOp::create(b, modType, a1Val);
          Value b2Mod = mod_arith::EncapsulateOp::create(b, modType, b2Val);
          Value a2Mod = mod_arith::EncapsulateOp::create(b, modType, a2Val);

          // out_bx = b1 * b2
          Value outBxMod =
              mod_arith::MontMulOp::create(b, modType, b1Mod, b2Mod);
          // out_rx = a1 * a2
          Value outRxMod =
              mod_arith::MontMulOp::create(b, modType, a1Mod, a2Mod);
          // out_ax = (b1 * a2) + (a1 * b2)
          Value p1Mod = mod_arith::MontMulOp::create(b, modType, b1Mod, a2Mod);
          Value p2Mod = mod_arith::MontMulOp::create(b, modType, a1Mod, b2Mod);
          Value outAxMod = mod_arith::AddOp::create(b, modType, p1Mod, p2Mod);

          Value outBx = mod_arith::LiftOp::create(
              b, elemType, outBxMod, mod_arith::LiftType::STANDARD);
          Value outAx = mod_arith::LiftOp::create(
              b, elemType, outAxMod, mod_arith::LiftType::STANDARD);
          Value outRx = mod_arith::LiftOp::create(
              b, elemType, outRxMod, mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, outBx, op.getOutBx(), ValueRange{lVal, i});
          memref::StoreOp::create(b, outAx, op.getOutAx(), ValueRange{lVal, i});
          memref::StoreOp::create(b, outRx, op.getOutRx(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 12. TensorSquareOp: Degree-1 ciphertext squaring
struct ConvertTensorSquareOp : public CheddarGPUOpConversion<TensorSquareOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(TensorSquareOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Type elemType = outType.getElementType();

          Value b1Val =
              memref::LoadOp::create(b, op.getSrcBx(), ValueRange{lVal, i});
          Value a1Val =
              memref::LoadOp::create(b, op.getSrcAx(), ValueRange{lVal, i});

          Value b1Mod = mod_arith::EncapsulateOp::create(b, modType, b1Val);
          Value a1Mod = mod_arith::EncapsulateOp::create(b, modType, a1Val);

          // out_bx = b1 * b1
          Value outBxMod =
              mod_arith::MontMulOp::create(b, modType, b1Mod, b1Mod);
          // out_rx = a1 * a1
          Value outRxMod =
              mod_arith::MontMulOp::create(b, modType, a1Mod, a1Mod);
          // out_ax = 2 * (b1 * a1) = (b1 * a1) + (b1 * a1)
          Value prodMod =
              mod_arith::MontMulOp::create(b, modType, b1Mod, a1Mod);
          Value outAxMod =
              mod_arith::AddOp::create(b, modType, prodMod, prodMod);

          Value outBx = mod_arith::LiftOp::create(
              b, elemType, outBxMod, mod_arith::LiftType::STANDARD);
          Value outAx = mod_arith::LiftOp::create(
              b, elemType, outAxMod, mod_arith::LiftType::STANDARD);
          Value outRx = mod_arith::LiftOp::create(
              b, elemType, outRxMod, mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, outBx, op.getOutBx(), ValueRange{lVal, i});
          memref::StoreOp::create(b, outAx, op.getOutAx(), ValueRange{lVal, i});
          memref::StoreOp::create(b, outRx, op.getOutRx(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 13. MulImaginaryUnitOp
struct ConvertMulImaginaryUnitOp
    : public CheddarGPUOpConversion<MulImaginaryUnitOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(MulImaginaryUnitOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    emitPerLimbCoeffLoop(
        op, rewriter, *modParams, lStart, lCount,
        [&](ImplicitLocOpBuilder& b, int64_t l, uint64_t /*modulus*/,
            mod_arith::ModArithType modType, Value i) {
          Value lVal = arith::ConstantIndexOp::create(b, l);
          Value inVal =
              memref::LoadOp::create(b, op.getIn(), ValueRange{lVal, i});
          Value psiVal =
              memref::LoadOp::create(b, op.getIUnit(), ValueRange{lVal});

          Value inMod = mod_arith::EncapsulateOp::create(b, modType, inVal);
          Value psiMod = mod_arith::EncapsulateOp::create(b, modType, psiVal);

          // In Cheddar:
          // if (i >= N/2): psi = q - psi (i.e. modular negation)
          int64_t halfDegree = modParams->degree / 2;
          Value cHalfDegree = arith::ConstantIndexOp::create(b, halfDegree);
          Value isSecondHalf = arith::CmpIOp::create(
              b, arith::CmpIPredicate::sge, i, cHalfDegree);

          Value zeroVal = arith::ConstantOp::create(
              b, inVal.getType(), b.getIntegerAttr(inVal.getType(), 0));
          Value zeroMod = mod_arith::EncapsulateOp::create(b, modType, zeroVal);
          Value negPsiMod =
              mod_arith::SubOp::create(b, modType, zeroMod, psiMod);

          Value negPsi = mod_arith::LiftOp::create(
              b, inVal.getType(), negPsiMod, mod_arith::LiftType::STANDARD);
          Value factorVal =
              arith::SelectOp::create(b, isSecondHalf, negPsi, psiVal);
          Value factorMod =
              mod_arith::EncapsulateOp::create(b, modType, factorVal);
          Value prodMod =
              mod_arith::MontMulOp::create(b, modType, inMod, factorMod);
          Value res = mod_arith::LiftOp::create(b, inVal.getType(), prodMod,
                                                mod_arith::LiftType::STANDARD);

          memref::StoreOp::create(b, res, op.getOut(), ValueRange{lVal, i});
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// 14. PermuteOp
struct ConvertPermuteOp : public CheddarGPUOpConversion<PermuteOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    uint64_t galois = 0;
    if (op.getConj()) {
      galois = 2 * modParams->degree - 1;
    } else if (op.getGalois().has_value()) {
      galois = *op.getGalois();
    } else {
      return op.emitOpError("permute missing galois factor");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value cDegree = arith::ConstantIndexOp::create(b, modParams->degree);
    Value c1 = arith::ConstantIndexOp::create(b, 1);

    // Permute maps destination index x to source index x_src:
    // out[l, x] = in[l, x_src]
    // Since x_src is purely a function of x and galois, we emit an
    // scf.index_switch or unrolled index mapping if small, or computed on index
    // using bit-reversal. For general degrees N (e.g. up to 65536), computing
    // the index formula per coefficient:
    for (int64_t l = lStart; l < lStart + lCount; ++l) {
      Value lVal = arith::ConstantIndexOp::create(b, l);
      scf::ForOp::create(
          b, c0, cDegree, c1, ValueRange{},
          [&](OpBuilder& nestedBuilder, Location loc, Value xIdx,
              ValueRange /*iterArgs*/) {
            ImplicitLocOpBuilder nb(loc, nestedBuilder);
            uint32_t bits = modParams->logN + 1;

            // In Cheddar GPU:
            // x_rev = bitReverse(x, log_n + 1) + 1
            // x_src = bitReverse((x_rev * galois - 1) % 2N, log_n + 1)
            // We can emit the bit reverse on index:
            // x_i32 = arith.index_cast xIdx : index to i32
            Value xI32 = arith::IndexCastOp::create(
                nb, IntegerType::get(nb.getContext(), 32), xIdx);

            auto emitBitReverse = [&](Value val, uint32_t numBits) -> Value {
              Type i32Ty = IntegerType::get(nb.getContext(), 32);
              Value zero32 = arith::ConstantIntOp::create(nb, i32Ty, 0);
              Value c0_idx = arith::ConstantIndexOp::create(nb, 0);
              Value cBits_idx = arith::ConstantIndexOp::create(nb, numBits);
              Value c1_idx = arith::ConstantIndexOp::create(nb, 1);

              auto forOp = scf::ForOp::create(
                  nb, c0_idx, cBits_idx, c1_idx, ValueRange{zero32},
                  [&](OpBuilder& b2, Location l2, Value bitIdx,
                      ValueRange iterArgs) {
                    ImplicitLocOpBuilder bLoop(l2, b2);
                    Value acc = iterArgs[0];
                    Value bitIdx32 =
                        arith::IndexCastOp::create(bLoop, i32Ty, bitIdx);
                    Value bitShifted =
                        arith::ShRUIOp::create(bLoop, val, bitIdx32);
                    Value bit = arith::AndIOp::create(
                        bLoop, bitShifted,
                        arith::ConstantIntOp::create(bLoop, i32Ty, 1));
                    Value targetShift = arith::SubIOp::create(
                        bLoop,
                        arith::ConstantIntOp::create(bLoop, i32Ty, numBits - 1),
                        bitIdx32);
                    Value revBit =
                        arith::ShLIOp::create(bLoop, bit, targetShift);
                    Value newAcc = arith::OrIOp::create(bLoop, acc, revBit);
                    scf::YieldOp::create(bLoop, newAcc);
                  });
              return forOp.getResult(0);
            };

            Value xRev = emitBitReverse(xI32, bits);
            Value one32 = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 32), 1);
            Value xRevPlus1 = arith::AddIOp::create(nb, xRev, one32);

            Value galoisVal = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 64), galois);
            Value xRevPlus1_64 = arith::ExtUIOp::create(
                nb, IntegerType::get(nb.getContext(), 64), xRevPlus1);
            Value prod64 = arith::MulIOp::create(nb, xRevPlus1_64, galoisVal);
            Value one64 = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 64), 1);
            Value prodMinus1 = arith::SubIOp::create(nb, prod64, one64);
            uint64_t mask = (1ULL << bits) - 1;
            Value maskVal = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 64), mask);
            Value mod2N = arith::AndIOp::create(nb, prodMinus1, maskVal);
            Value mod2N_32 = arith::TruncIOp::create(
                nb, IntegerType::get(nb.getContext(), 32), mod2N);

            Value xSrc32 = emitBitReverse(mod2N_32, bits);
            Value xSrcIdx = arith::IndexCastOp::create(
                nb, IndexType::get(nb.getContext()), xSrc32);

            Value val = memref::LoadOp::create(nb, op.getIn(),
                                               ValueRange{lVal, xSrcIdx});
            memref::StoreOp::create(nb, val, op.getOut(),
                                    ValueRange{lVal, xIdx});
            scf::YieldOp::create(nb);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// 14b. Permute2Op
struct ConvertPermute2Op : public CheddarGPUOpConversion<Permute2Op> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(Permute2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    uint64_t galois = 0;
    if (op.getConj()) {
      galois = 2 * modParams->degree - 1;
    } else if (op.getGalois().has_value()) {
      galois = *op.getGalois();
    } else {
      return op.emitOpError("permute_2 missing galois factor");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value cDegree = arith::ConstantIndexOp::create(b, modParams->degree);
    Value c1 = arith::ConstantIndexOp::create(b, 1);

    for (int64_t l = lStart; l < lStart + lCount; ++l) {
      Value lVal = arith::ConstantIndexOp::create(b, l);
      scf::ForOp::create(
          b, c0, cDegree, c1, ValueRange{},
          [&](OpBuilder& nestedBuilder, Location loc, Value xIdx,
              ValueRange /*iterArgs*/) {
            ImplicitLocOpBuilder nb(loc, nestedBuilder);
            uint32_t bits = modParams->logN + 1;

            Value xI32 = arith::IndexCastOp::create(
                nb, IntegerType::get(nb.getContext(), 32), xIdx);

            auto emitBitReverse = [&](Value val, uint32_t numBits) -> Value {
              Type i32Ty = IntegerType::get(nb.getContext(), 32);
              Value zero32 = arith::ConstantIntOp::create(nb, i32Ty, 0);
              Value c0_idx = arith::ConstantIndexOp::create(nb, 0);
              Value cBits_idx = arith::ConstantIndexOp::create(nb, numBits);
              Value c1_idx = arith::ConstantIndexOp::create(nb, 1);

              auto forOp = scf::ForOp::create(
                  nb, c0_idx, cBits_idx, c1_idx, ValueRange{zero32},
                  [&](OpBuilder& b2, Location l2, Value bitIdx,
                      ValueRange iterArgs) {
                    ImplicitLocOpBuilder bLoop(l2, b2);
                    Value acc = iterArgs[0];
                    Value bitIdx32 =
                        arith::IndexCastOp::create(bLoop, i32Ty, bitIdx);
                    Value bitShifted =
                        arith::ShRUIOp::create(bLoop, val, bitIdx32);
                    Value bit = arith::AndIOp::create(
                        bLoop, bitShifted,
                        arith::ConstantIntOp::create(bLoop, i32Ty, 1));
                    Value targetShift = arith::SubIOp::create(
                        bLoop,
                        arith::ConstantIntOp::create(bLoop, i32Ty, numBits - 1),
                        bitIdx32);
                    Value revBit =
                        arith::ShLIOp::create(bLoop, bit, targetShift);
                    Value newAcc = arith::OrIOp::create(bLoop, acc, revBit);
                    scf::YieldOp::create(bLoop, newAcc);
                  });
              return forOp.getResult(0);
            };

            Value xRev = emitBitReverse(xI32, bits);
            Value one32 = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 32), 1);
            Value xRevPlus1 = arith::AddIOp::create(nb, xRev, one32);

            Value galoisVal = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 64), galois);
            Value xRevPlus1_64 = arith::ExtUIOp::create(
                nb, IntegerType::get(nb.getContext(), 64), xRevPlus1);
            Value prod64 = arith::MulIOp::create(nb, xRevPlus1_64, galoisVal);
            Value one64 = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 64), 1);
            Value prodMinus1 = arith::SubIOp::create(nb, prod64, one64);
            uint64_t mask = (1ULL << bits) - 1;
            Value maskVal = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 64), mask);
            Value mod2N = arith::AndIOp::create(nb, prodMinus1, maskVal);
            Value mod2N_32 = arith::TruncIOp::create(
                nb, IntegerType::get(nb.getContext(), 32), mod2N);

            Value xSrc32 = emitBitReverse(mod2N_32, bits);
            Value xSrcIdx = arith::IndexCastOp::create(
                nb, IndexType::get(nb.getContext()), xSrc32);

            Value valBx = memref::LoadOp::create(nb, op.getInBx(),
                                                 ValueRange{lVal, xSrcIdx});
            Value valAx = memref::LoadOp::create(nb, op.getInAx(),
                                                 ValueRange{lVal, xSrcIdx});
            memref::StoreOp::create(nb, valBx, op.getOutBx(),
                                    ValueRange{lVal, xIdx});
            memref::StoreOp::create(nb, valAx, op.getOutAx(),
                                    ValueRange{lVal, xIdx});
            scf::YieldOp::create(nb);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// 15. PermuteAccumOp
struct ConvertPermuteAccumOp : public CheddarGPUOpConversion<PermuteAccumOp> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(PermuteAccumOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Type intType = IntegerType::get(b.getContext(), modParams->word);

    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value cDegree = arith::ConstantIndexOp::create(b, modParams->degree);
    Value c1 = arith::ConstantIndexOp::create(b, 1);

    auto emitBitReverse = [&](ImplicitLocOpBuilder& nb, Value val,
                              uint32_t numBits) -> Value {
      Type i32Ty = IntegerType::get(nb.getContext(), 32);
      Value zero32 = arith::ConstantIntOp::create(nb, i32Ty, 0);
      Value c0_idx = arith::ConstantIndexOp::create(nb, 0);
      Value cBits_idx = arith::ConstantIndexOp::create(nb, numBits);
      Value c1_idx = arith::ConstantIndexOp::create(nb, 1);

      auto forOp = scf::ForOp::create(
          nb, c0_idx, cBits_idx, c1_idx, ValueRange{zero32},
          [&](OpBuilder& b2, Location l2, Value bitIdx, ValueRange iterArgs) {
            ImplicitLocOpBuilder bLoop(l2, b2);
            Value acc = iterArgs[0];
            Value bitIdx32 = arith::IndexCastOp::create(bLoop, i32Ty, bitIdx);
            Value bitShifted = arith::ShRUIOp::create(bLoop, val, bitIdx32);
            Value bit = arith::AndIOp::create(
                bLoop, bitShifted,
                arith::ConstantIntOp::create(bLoop, i32Ty, 1));
            Value targetShift = arith::SubIOp::create(
                bLoop, arith::ConstantIntOp::create(bLoop, i32Ty, numBits - 1),
                bitIdx32);
            Value revBit = arith::ShLIOp::create(bLoop, bit, targetShift);
            Value newAcc = arith::OrIOp::create(bLoop, acc, revBit);
            scf::YieldOp::create(bLoop, newAcc);
          });
      return forOp.getResult(0);
    };

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t primeOffset = op.getPrimeOffset().value_or(lStart);
    int64_t numQ = lCount - numAux;
    int64_t totalPrimes = modParams->params.getNumLimbs();
    int64_t alpha = modParams->params.getAux().size();
    int64_t auxStart = totalPrimes - alpha;

    for (int64_t l = lStart; l < lStart + lCount; ++l) {
      int64_t localLimb = l - lStart;
      bool isAux = (numAux > 0 && localLimb >= numQ);
      int64_t absLimb =
          isAux ? ((localLimb - numQ) + auxStart) : (localLimb + primeOffset);
      uint64_t q = modParams->params.getPrimeAt(absLimb);
      auto modAttr = IntegerAttr::get(intType, APInt(modParams->word, q));
      auto modType = mod_arith::ModArithType::get(b.getContext(), modAttr);
      Value lVal = arith::ConstantIndexOp::create(b, l);

      scf::ForOp::create(
          b, c0, cDegree, c1, ValueRange{},
          [&](OpBuilder& nestedBuilder, Location loc, Value xIdx,
              ValueRange /*iterArgs*/) {
            ImplicitLocOpBuilder nb(loc, nestedBuilder);
            uint32_t bits = modParams->logN + 1;

            Value xI32 = arith::IndexCastOp::create(
                nb, IntegerType::get(nb.getContext(), 32), xIdx);
            Value xRev = emitBitReverse(nb, xI32, bits);
            Value one32 = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 32), 1);
            Value xRevPlus1 = arith::AddIOp::create(nb, xRev, one32);
            Value xRevPlus1_64 = arith::ExtUIOp::create(
                nb, IntegerType::get(nb.getContext(), 64), xRevPlus1);
            uint64_t mask = (1ULL << bits) - 1;
            Value maskVal = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 64), mask);
            Value one64 = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 64), 1);

            Value accMod = nullptr;
            if (op.getAcc()) {
              Value baseVal = memref::LoadOp::create(nb, op.getAcc(),
                                                     ValueRange{lVal, xIdx});
              accMod = mod_arith::EncapsulateOp::create(nb, modType, baseVal);
            }

            for (size_t k = 0; k < op.getSrcs().size(); ++k) {
              uint64_t galois = op.getGaloisFactors()[k];
              Value galoisVal = arith::ConstantIntOp::create(
                  nb, IntegerType::get(nb.getContext(), 64), galois);
              Value prod64 = arith::MulIOp::create(nb, xRevPlus1_64, galoisVal);
              Value prodMinus1 = arith::SubIOp::create(nb, prod64, one64);
              Value mod2N = arith::AndIOp::create(nb, prodMinus1, maskVal);
              Value mod2N_32 = arith::TruncIOp::create(
                  nb, IntegerType::get(nb.getContext(), 32), mod2N);
              Value xSrc32 = emitBitReverse(nb, mod2N_32, bits);
              Value xSrcIdx = arith::IndexCastOp::create(
                  nb, IndexType::get(nb.getContext()), xSrc32);

              Value srcVal = memref::LoadOp::create(nb, op.getSrcs()[k],
                                                    ValueRange{lVal, xSrcIdx});
              Value srcMod =
                  mod_arith::EncapsulateOp::create(nb, modType, srcVal);

              if (!accMod) {
                accMod = srcMod;
              } else {
                accMod = mod_arith::AddOp::create(nb, modType, accMod, srcMod);
              }
            }

            Value res =
                mod_arith::LiftOp::create(nb, outType.getElementType(), accMod,
                                          mod_arith::LiftType::STANDARD);
            memref::StoreOp::create(nb, res, op.getOut(),
                                    ValueRange{lVal, xIdx});
            scf::YieldOp::create(nb);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// 15b. PermuteAccum2Op
struct ConvertPermuteAccum2Op : public CheddarGPUOpConversion<PermuteAccum2Op> {
  using CheddarGPUOpConversion::CheddarGPUOpConversion;

  LogicalResult matchAndRewrite(PermuteAccum2Op op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams)
      return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Type intType = IntegerType::get(b.getContext(), modParams->word);

    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value cDegree = arith::ConstantIndexOp::create(b, modParams->degree);
    Value c1 = arith::ConstantIndexOp::create(b, 1);

    auto emitBitReverse = [&](ImplicitLocOpBuilder& nb, Value val,
                              uint32_t numBits) -> Value {
      Type i32Ty = IntegerType::get(nb.getContext(), 32);
      Value zero32 = arith::ConstantIntOp::create(nb, i32Ty, 0);
      Value c0_idx = arith::ConstantIndexOp::create(nb, 0);
      Value cBits_idx = arith::ConstantIndexOp::create(nb, numBits);
      Value c1_idx = arith::ConstantIndexOp::create(nb, 1);

      auto forOp = scf::ForOp::create(
          nb, c0_idx, cBits_idx, c1_idx, ValueRange{zero32},
          [&](OpBuilder& b2, Location l2, Value bitIdx, ValueRange iterArgs) {
            ImplicitLocOpBuilder bLoop(l2, b2);
            Value acc = iterArgs[0];
            Value bitIdx32 = arith::IndexCastOp::create(bLoop, i32Ty, bitIdx);
            Value bitShifted = arith::ShRUIOp::create(bLoop, val, bitIdx32);
            Value bit = arith::AndIOp::create(
                bLoop, bitShifted,
                arith::ConstantIntOp::create(bLoop, i32Ty, 1));
            Value targetShift = arith::SubIOp::create(
                bLoop, arith::ConstantIntOp::create(bLoop, i32Ty, numBits - 1),
                bitIdx32);
            Value revBit = arith::ShLIOp::create(bLoop, bit, targetShift);
            Value newAcc = arith::OrIOp::create(bLoop, acc, revBit);
            scf::YieldOp::create(bLoop, newAcc);
          });
      return forOp.getResult(0);
    };

    int64_t numAux = op.getNumAux().value_or(0);
    int64_t primeOffset = op.getPrimeOffset().value_or(lStart);
    int64_t numQ = lCount - numAux;
    int64_t totalPrimes = modParams->params.getNumLimbs();
    int64_t alpha = modParams->params.getAux().size();
    int64_t auxStart = totalPrimes - alpha;

    for (int64_t l = lStart; l < lStart + lCount; ++l) {
      int64_t localLimb = l - lStart;
      bool isAux = (numAux > 0 && localLimb >= numQ);
      int64_t absLimb =
          isAux ? ((localLimb - numQ) + auxStart) : (localLimb + primeOffset);
      uint64_t q = modParams->params.getPrimeAt(absLimb);
      auto modAttr = IntegerAttr::get(intType, APInt(modParams->word, q));
      auto modType = mod_arith::ModArithType::get(b.getContext(), modAttr);
      Value lVal = arith::ConstantIndexOp::create(b, l);

      scf::ForOp::create(
          b, c0, cDegree, c1, ValueRange{},
          [&](OpBuilder& nestedBuilder, Location loc, Value xIdx,
              ValueRange /*iterArgs*/) {
            ImplicitLocOpBuilder nb(loc, nestedBuilder);
            uint32_t bits = modParams->logN + 1;

            Value xI32 = arith::IndexCastOp::create(
                nb, IntegerType::get(nb.getContext(), 32), xIdx);
            Value xRev = emitBitReverse(nb, xI32, bits);
            Value one32 = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 32), 1);
            Value xRevPlus1 = arith::AddIOp::create(nb, xRev, one32);
            Value xRevPlus1_64 = arith::ExtUIOp::create(
                nb, IntegerType::get(nb.getContext(), 64), xRevPlus1);
            uint64_t mask = (1ULL << bits) - 1;
            Value maskVal = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 64), mask);
            Value one64 = arith::ConstantIntOp::create(
                nb, IntegerType::get(nb.getContext(), 64), 1);

            Value accBxMod = nullptr;
            Value accAxMod = nullptr;
            if (op.getAccBx()) {
              Value baseBxVal = memref::LoadOp::create(nb, op.getAccBx(),
                                                       ValueRange{lVal, xIdx});
              accBxMod =
                  mod_arith::EncapsulateOp::create(nb, modType, baseBxVal);
              Value baseAxVal = memref::LoadOp::create(nb, op.getAccAx(),
                                                       ValueRange{lVal, xIdx});
              accAxMod =
                  mod_arith::EncapsulateOp::create(nb, modType, baseAxVal);
            }

            for (size_t k = 0; k < op.getSrcsBx().size(); ++k) {
              uint64_t galois = op.getGaloisFactors()[k];
              Value galoisVal = arith::ConstantIntOp::create(
                  nb, IntegerType::get(nb.getContext(), 64), galois);
              Value prod64 = arith::MulIOp::create(nb, xRevPlus1_64, galoisVal);
              Value prodMinus1 = arith::SubIOp::create(nb, prod64, one64);
              Value mod2N = arith::AndIOp::create(nb, prodMinus1, maskVal);
              Value mod2N_32 = arith::TruncIOp::create(
                  nb, IntegerType::get(nb.getContext(), 32), mod2N);
              Value xSrc32 = emitBitReverse(nb, mod2N_32, bits);
              Value xSrcIdx = arith::IndexCastOp::create(
                  nb, IndexType::get(nb.getContext()), xSrc32);

              Value srcBxVal = memref::LoadOp::create(
                  nb, op.getSrcsBx()[k], ValueRange{lVal, xSrcIdx});
              Value srcBxMod =
                  mod_arith::EncapsulateOp::create(nb, modType, srcBxVal);
              Value srcAxVal = memref::LoadOp::create(
                  nb, op.getSrcsAx()[k], ValueRange{lVal, xSrcIdx});
              Value srcAxMod =
                  mod_arith::EncapsulateOp::create(nb, modType, srcAxVal);

              if (!accBxMod) {
                accBxMod = srcBxMod;
                accAxMod = srcAxMod;
              } else {
                accBxMod =
                    mod_arith::AddOp::create(nb, modType, accBxMod, srcBxMod);
                accAxMod =
                    mod_arith::AddOp::create(nb, modType, accAxMod, srcAxMod);
              }
            }

            Value resBx = mod_arith::LiftOp::create(
                nb, outType.getElementType(), accBxMod,
                mod_arith::LiftType::STANDARD);
            Value resAx = mod_arith::LiftOp::create(
                nb, outType.getElementType(), accAxMod,
                mod_arith::LiftType::STANDARD);
            memref::StoreOp::create(nb, resBx, op.getOutBx(),
                                    ValueRange{lVal, xIdx});
            memref::StoreOp::create(nb, resAx, op.getOutAx(),
                                    ValueRange{lVal, xIdx});
            scf::YieldOp::create(nb);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct CheddarGPUToSCF : impl::CheddarGPUToSCFBase<CheddarGPUToSCF> {
  using CheddarGPUToSCFBase::CheddarGPUToSCFBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    ModuleOp module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<ConvertCopyOp, ConvertAddOp, ConvertSubOp, ConvertNegOp,
                 ConvertAdd2Op, ConvertSub2Op, ConvertNeg2Op, ConvertAddConstOp,
                 ConvertSubConstOp, ConvertSubOppositeConstOp, ConvertMulOp,
                 ConvertMulConstOp, ConvertCPAccumOp, ConvertCPAccumAddOp,
                 ConvertCPAccum2Op, ConvertCPAccumAdd2Op, ConvertTensorOp,
                 ConvertTensorSquareOp, ConvertMulImaginaryUnitOp,
                 ConvertPermuteOp, ConvertPermute2Op, ConvertPermuteAccumOp,
                 ConvertPermuteAccum2Op, ConvertZeroPadOp, ConvertModUpToMaxOp>(
        context);
    populateBConvToSCFPatterns(patterns);
    populateNTTToSCFPatterns(patterns);
    populateModDownEpilogueToSCFPatterns(patterns);
    populateHoistToSCFPatterns(patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           scf::SCFDialect, mod_arith::ModArithDialect>();
    target.addIllegalOp<
        CopyOp, AddOp, SubOp, NegOp, Add2Op, Sub2Op, Neg2Op, AddConstOp,
        SubConstOp, SubOppositeConstOp, MulOp, MulConstOp, CPAccumOp,
        CPAccumAddOp, CPAccum2Op, CPAccumAdd2Op, TensorOp, TensorSquareOp,
        MulImaginaryUnitOp, PermuteOp, Permute2Op, PermuteAccumOp,
        PermuteAccum2Op, ZeroPadOp, ModUpToMaxOp, BConvOp, NTTOp, INTTOp,
        ModDownEpilogueOp, NTTModDownOp>();
    configureHoistSCFConversionTarget(target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Strip module attribute so downstream tools without CheddarGPUDialect
    // (like mlir-translate) don't error
    module->removeAttr("cheddar_gpu.params");
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCheddarGPUToSCFPass() {
  return std::make_unique<CheddarGPUToSCF>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
