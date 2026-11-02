#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToSCF/HoistPatterns.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "llvm/include/llvm/ADT/APInt.h"                 // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
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
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

namespace {

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

static Value buildBitReverse(ImplicitLocOpBuilder& b, Value val,
                             uint32_t numBits) {
  Type i32Ty = IntegerType::get(b.getContext(), 32);
  Value v = val;
  auto swapBits = [&](uint32_t mask, uint32_t shift) {
    Value m = arith::ConstantIntOp::create(b, i32Ty, mask);
    Value s_r = arith::ShRUIOp::create(
        b, v, arith::ConstantIntOp::create(b, i32Ty, shift));
    Value a_r = arith::AndIOp::create(b, s_r, m);
    Value a_l = arith::AndIOp::create(b, v, m);
    Value s_l = arith::ShLIOp::create(
        b, a_l, arith::ConstantIntOp::create(b, i32Ty, shift));
    v = arith::OrIOp::create(b, a_r, s_l);
  };
  swapBits(0x55555555, 1);
  swapBits(0x33333333, 2);
  swapBits(0x0F0F0F0F, 4);
  swapBits(0x00FF00FF, 8);
  swapBits(0x0000FFFF, 16);
  if (32 > numBits) {
    v = arith::ShRUIOp::create(
        b, v, arith::ConstantIntOp::create(b, i32Ty, 32 - numBits));
  }
  return v;
}

// === ConvertHoistBSFusedOp ===

struct ConvertHoistBSFusedOp : public OpRewritePattern<HoistBSFusedOp> {
  using OpRewritePattern<HoistBSFusedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HoistBSFusedOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams) {
      return op.emitOpError("missing cheddar_gpu.params on module");
    }

    uint32_t word = modParams->word;
    int64_t degree = modParams->degree;
    uint32_t logN = modParams->logN;
    size_t R = op.getOutBx().size();
    size_t beta = op.getModUp().size();

    auto outType = ::llvm::cast<MemRefType>(op.getOutBx()[0].getType());
    int64_t numLimbs = outType.getDimSize(0);
    int64_t numQ = op.getNumQ();
    int64_t numAux = op.getNumAux();
    int64_t primeOffset = op.getPrimeOffset();
    int64_t totalPrimes = modParams->params.getNumLimbs();
    int64_t alpha = modParams->params.getAux().size();
    int64_t auxStart = totalPrimes - alpha;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Type intType = IntegerType::get(b.getContext(), word);
    Type i32Ty = b.getI32Type();
    Type i64Ty = b.getI64Type();
    auto dynMemref1DWordType = MemRefType::get({ShapedType::kDynamic}, intType);

    // Compute EVK word offsets
    SmallVector<int64_t> evkWordOffsets;
    ArrayRef<int64_t> galois = op.getGaloisFactors();
    ArrayRef<int64_t> evkNumQ = op.getEvkNumQ();
    ArrayRef<int64_t> evkByteOffsets = op.getEvkByteOffsets();
    int64_t wordBytes = word / 8;

    if (evkByteOffsets.size() == R) {
      for (size_t k = 0; k < R; ++k) {
        int64_t baseWord = evkByteOffsets[k] / wordBytes;
        int64_t lk = evkNumQ[k] + numAux;
        int64_t polyStride = lk * degree;
        for (size_t j = 0; j < beta; ++j) {
          evkWordOffsets.push_back(baseWord + (j * 2 + 0) * polyStride);
          evkWordOffsets.push_back(baseWord + (j * 2 + 1) * polyStride);
        }
      }
    } else {
      for (size_t k = 0; k < R; ++k) {
        for (size_t j = 0; j < beta; ++j) {
          size_t bxIdx = (k * beta + j) * 2 + 0;
          size_t axIdx = (k * beta + j) * 2 + 1;
          evkWordOffsets.push_back(evkByteOffsets[bxIdx] / wordBytes);
          evkWordOffsets.push_back(evkByteOffsets[axIdx] / wordBytes);
        }
      }
    }

    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value sizeBytes = memref::DimOp::create(b, op.getEvkSlab(), c0);
    Value wordBytesVal = arith::ConstantIndexOp::create(b, wordBytes);
    Value sizeWords = arith::DivUIOp::create(b, sizeBytes, wordBytesVal);
    Value evkSlabWords = memref::ViewOp::create(
        b, dynMemref1DWordType, op.getEvkSlab(), c0, ValueRange{sizeWords});

    Value cDegree = arith::ConstantIndexOp::create(b, degree);
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    uint32_t bits = logN + 1;

    for (int64_t l = 0; l < numLimbs; ++l) {
      bool isAux = (numAux > 0 && l >= numQ);
      int64_t absLimb = isAux ? ((l - numQ) + auxStart) : (l + primeOffset);
      uint64_t prime = modParams->params.getPrimeAt(absLimb);
      auto modAttr = mod_arith::ModArithType::get(
          b.getContext(), IntegerAttr::get(intType, prime));
      Value lVal = arith::ConstantIndexOp::create(b, l);

      scf::ForOp::create(
          b, c0, cDegree, c1, ValueRange{},
          [&](OpBuilder& nestedBuilder, Location loc, Value xIdx,
              ValueRange /*iterArgs*/) {
            ImplicitLocOpBuilder nb(loc, nestedBuilder);

            SmallVector<Value> modUpVals;
            for (size_t j = 0; j < beta; ++j) {
              Value loaded = memref::LoadOp::create(nb, op.getModUp()[j],
                                                    ValueRange{lVal, xIdx});
              modUpVals.push_back(
                  mod_arith::EncapsulateOp::create(nb, modAttr, loaded));
            }

            Value inBxMod = nullptr;
            if (l < numQ) {
              Value inBxVal = memref::LoadOp::create(
                  nb, op.getInBxPseudoModup(), ValueRange{lVal, xIdx});
              inBxMod = mod_arith::EncapsulateOp::create(nb, modAttr, inBxVal);
            }

            Value xI32 = arith::IndexCastOp::create(nb, i32Ty, xIdx);
            Value xRev = buildBitReverse(nb, xI32, bits);
            Value one32 = arith::ConstantIntOp::create(nb, i32Ty, 1);
            Value xRevPlus1 = arith::AddIOp::create(nb, xRev, one32);
            Value xRevPlus1_64 = arith::ExtUIOp::create(nb, i64Ty, xRevPlus1);
            Value maskVal =
                arith::ConstantIntOp::create(nb, i64Ty, (1ULL << bits) - 1);
            Value one64 = arith::ConstantIntOp::create(nb, i64Ty, 1);

            for (size_t k = 0; k < R; ++k) {
              Value galoisVal =
                  arith::ConstantIntOp::create(nb, i64Ty, galois[k]);
              Value prod64 = arith::MulIOp::create(nb, xRevPlus1_64, galoisVal);
              Value prodMinus1 = arith::SubIOp::create(nb, prod64, one64);
              Value mod2N = arith::AndIOp::create(nb, prodMinus1, maskVal);
              Value mod2N_32 = arith::TruncIOp::create(nb, i32Ty, mod2N);
              Value dst32 = buildBitReverse(nb, mod2N_32, bits);
              Value dstIdx =
                  arith::IndexCastOp::create(nb, nb.getIndexType(), dst32);

              int64_t keyLimb = (l < numQ) ? l : (evkNumQ[k] + (l - numQ));

              Value accBxMod = nullptr;
              Value accAxMod = nullptr;

              for (size_t j = 0; j < beta; ++j) {
                int64_t baseBx = evkWordOffsets[(k * beta + j) * 2 + 0];
                int64_t baseAx = evkWordOffsets[(k * beta + j) * 2 + 1];
                Value offsetBx =
                    arith::AddIOp::create(nb,
                                          arith::ConstantIndexOp::create(
                                              nb, baseBx + keyLimb * degree),
                                          xIdx);
                Value offsetAx =
                    arith::AddIOp::create(nb,
                                          arith::ConstantIndexOp::create(
                                              nb, baseAx + keyLimb * degree),
                                          xIdx);

                Value keyBxVal = memref::LoadOp::create(nb, evkSlabWords,
                                                        ValueRange{offsetBx});
                Value keyAxVal = memref::LoadOp::create(nb, evkSlabWords,
                                                        ValueRange{offsetAx});

                Value keyBxMod =
                    mod_arith::EncapsulateOp::create(nb, modAttr, keyBxVal);
                Value keyAxMod =
                    mod_arith::EncapsulateOp::create(nb, modAttr, keyAxVal);

                Value multBx = mod_arith::MontMulOp::create(
                    nb, modAttr, modUpVals[j], keyBxMod);
                accBxMod = accBxMod ? mod_arith::AddOp::create(nb, modAttr,
                                                               accBxMod, multBx)
                                    : multBx;

                Value multAx = mod_arith::MontMulOp::create(
                    nb, modAttr, modUpVals[j], keyAxMod);
                accAxMod = accAxMod ? mod_arith::AddOp::create(nb, modAttr,
                                                               accAxMod, multAx)
                                    : multAx;
              }

              if (l < numQ) {
                accBxMod =
                    mod_arith::AddOp::create(nb, modAttr, accBxMod, inBxMod);
              }

              Value resBx = mod_arith::LiftOp::create(
                  nb, intType, accBxMod, mod_arith::LiftType::STANDARD);
              Value resAx = mod_arith::LiftOp::create(
                  nb, intType, accAxMod, mod_arith::LiftType::STANDARD);

              memref::StoreOp::create(nb, resBx, op.getOutBx()[k],
                                      ValueRange{lVal, dstIdx});
              memref::StoreOp::create(nb, resAx, op.getOutAx()[k],
                                      ValueRange{lVal, dstIdx});
            }
            scf::YieldOp::create(nb);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// === ConvertHoistGSFusedOp ===

struct ConvertHoistGSFusedOp : public OpRewritePattern<HoistGSFusedOp> {
  using OpRewritePattern<HoistGSFusedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HoistGSFusedOp op,
                                PatternRewriter& rewriter) const override {
    auto modParams = getModuleParams(op);
    if (!modParams) {
      return op.emitOpError("missing cheddar_gpu.params on module");
    }

    uint32_t word = modParams->word;
    int64_t degree = modParams->degree;
    size_t G = op.getOutBx().size();
    size_t B = op.getInBx().size();

    auto outType = ::llvm::cast<MemRefType>(op.getOutBx()[0].getType());
    int64_t numLimbs = outType.getDimSize(0);
    int64_t numQ = op.getNumQ();
    int64_t numAux = op.getNumAux();
    int64_t primeOffset = op.getPrimeOffset();
    int64_t totalPrimes = modParams->params.getNumLimbs();
    int64_t alpha = modParams->params.getAux().size();
    int64_t auxStart = totalPrimes - alpha;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Type intType = IntegerType::get(b.getContext(), word);
    auto dynMemref1DWordType = MemRefType::get({ShapedType::kDynamic}, intType);
    int64_t wordBytes = word / 8;

    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value sizeBytes = memref::DimOp::create(b, op.getConstsSlab(), c0);
    Value wordBytesVal = arith::ConstantIndexOp::create(b, wordBytes);
    Value sizeWords = arith::DivUIOp::create(b, sizeBytes, wordBytesVal);
    Value constsSlabWords = memref::ViewOp::create(
        b, dynMemref1DWordType, op.getConstsSlab(), c0, ValueRange{sizeWords});

    ArrayRef<int64_t> ptOffsets = op.getPtWordOffsets();
    Value cDegree = arith::ConstantIndexOp::create(b, degree);
    Value c1 = arith::ConstantIndexOp::create(b, 1);

    for (int64_t l = 0; l < numLimbs; ++l) {
      bool isAux = (numAux > 0 && l >= numQ);
      int64_t absLimb = isAux ? ((l - numQ) + auxStart) : (l + primeOffset);
      uint64_t prime = modParams->params.getPrimeAt(absLimb);
      auto modAttr = mod_arith::ModArithType::get(
          b.getContext(), IntegerAttr::get(intType, prime));
      Value lVal = arith::ConstantIndexOp::create(b, l);

      scf::ForOp::create(
          b, c0, cDegree, c1, ValueRange{},
          [&](OpBuilder& nestedBuilder, Location loc, Value xIdx,
              ValueRange /*iterArgs*/) {
            ImplicitLocOpBuilder nb(loc, nestedBuilder);

            SmallVector<Value> inBxMods;
            SmallVector<Value> inAxMods;
            for (size_t b_idx = 0; b_idx < B; ++b_idx) {
              Value bxVal = memref::LoadOp::create(nb, op.getInBx()[b_idx],
                                                   ValueRange{lVal, xIdx});
              Value axVal = memref::LoadOp::create(nb, op.getInAx()[b_idx],
                                                   ValueRange{lVal, xIdx});
              inBxMods.push_back(
                  mod_arith::EncapsulateOp::create(nb, modAttr, bxVal));
              inAxMods.push_back(
                  mod_arith::EncapsulateOp::create(nb, modAttr, axVal));
            }

            for (size_t g = 0; g < G; ++g) {
              Value accBxMod = nullptr;
              Value accAxMod = nullptr;

              for (size_t b_idx = 0; b_idx < B; ++b_idx) {
                int64_t ptOffset = ptOffsets[g * B + b_idx];
                if (ptOffset == -1) continue;

                Value offsetVal = arith::AddIOp::create(
                    nb,
                    arith::ConstantIndexOp::create(nb, ptOffset + l * degree),
                    xIdx);
                Value ptVal = memref::LoadOp::create(nb, constsSlabWords,
                                                     ValueRange{offsetVal});
                Value ptMod =
                    mod_arith::EncapsulateOp::create(nb, modAttr, ptVal);

                Value multBx = mod_arith::MontMulOp::create(
                    nb, modAttr, inBxMods[b_idx], ptMod);
                accBxMod = accBxMod ? mod_arith::AddOp::create(nb, modAttr,
                                                               accBxMod, multBx)
                                    : multBx;

                Value multAx = mod_arith::MontMulOp::create(
                    nb, modAttr, inAxMods[b_idx], ptMod);
                accAxMod = accAxMod ? mod_arith::AddOp::create(nb, modAttr,
                                                               accAxMod, multAx)
                                    : multAx;
              }

              Value zeroWord = arith::ConstantIntOp::create(nb, intType, 0);
              Value resBx = accBxMod ? mod_arith::LiftOp::create(
                                           nb, intType, accBxMod,
                                           mod_arith::LiftType::STANDARD)
                                     : zeroWord;
              Value resAx = accAxMod ? mod_arith::LiftOp::create(
                                           nb, intType, accAxMod,
                                           mod_arith::LiftType::STANDARD)
                                     : zeroWord;

              memref::StoreOp::create(nb, resBx, op.getOutBx()[g],
                                      ValueRange{lVal, xIdx});
              memref::StoreOp::create(nb, resAx, op.getOutAx()[g],
                                      ValueRange{lVal, xIdx});
            }
            scf::YieldOp::create(nb);
          });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateHoistToSCFPatterns(RewritePatternSet& patterns) {
  patterns.add<ConvertHoistBSFusedOp, ConvertHoistGSFusedOp>(
      patterns.getContext());
}

void configureHoistSCFConversionTarget(ConversionTarget& target) {
  target.addIllegalOp<HoistBSFusedOp, HoistGSFusedOp>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
