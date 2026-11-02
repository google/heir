#include "lib/Dialect/CheddarGPU/Conversions/CheddarGPUToSCF/NTTPatterns.h"

#include <cstdint>
#include <string>
#include <utility>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/Tables.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

namespace {

std::pair<int64_t, int64_t> getLimbRange(std::optional<ArrayRef<int64_t>> range,
                                         int64_t totalLimbs) {
  if (range && range->size() == 2) {
    return {(*range)[0], (*range)[1]};
  }
  return {0, totalLimbs};
}

template <unsigned W>
cheddar_ref::Params<W> paramsFromAttr(ParamsAttr attr) {
  cheddar_ref::Params<W> p;
  p.word_size = W;
  p.log_n = attr.getLogN();
  p.degree = 1ULL << p.log_n;
  p.alpha = attr.getAlpha();
  p.dnum = attr.getDnum();
  for (int64_t v : attr.getMain().asArrayRef()) {
    p.main_primes.push_back(
        static_cast<typename cheddar_ref::Params<W>::Word>(v));
  }
  for (int64_t v : attr.getAux().asArrayRef()) {
    p.aux_primes.push_back(
        static_cast<typename cheddar_ref::Params<W>::Word>(v));
  }
  for (int64_t v : attr.getTer().asArrayRef()) {
    p.ter_primes.push_back(
        static_cast<typename cheddar_ref::Params<W>::Word>(v));
  }
  size_t num_ter = p.ter_primes.size();
  for (size_t i = 0; i < num_ter; ++i) {
    p.all_primes.push_back(p.ter_primes[num_ter - 1 - i]);
  }
  for (auto prime : p.main_primes) p.all_primes.push_back(prime);
  for (auto prime : p.aux_primes) p.all_primes.push_back(prime);
  return p;
}

template <typename Word>
Value getOrCreateTwiddleGlobal(ImplicitLocOpBuilder& b, ModuleOp module,
                               StringRef prefix, uint64_t q, size_t degree,
                               Type intType, const Word* twiddle_data) {
  std::string name =
      (prefix + "_" + std::to_string(q) + "_" + std::to_string(degree)).str();
  auto globalOp = module.lookupSymbol<memref::GlobalOp>(name);
  if (!globalOp) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    SmallVector<APInt> twiddles;
    twiddles.reserve(degree);
    unsigned width = intType.getIntOrFloatBitWidth();
    for (size_t i = 0; i < degree; ++i) {
      twiddles.push_back(APInt(width, twiddle_data[i]));
    }
    auto tensorType =
        RankedTensorType::get({static_cast<int64_t>(degree)}, intType);
    auto denseAttr = DenseElementsAttr::get(tensorType, twiddles);
    auto globalMemrefType =
        MemRefType::get({static_cast<int64_t>(degree)}, intType);
    globalOp = memref::GlobalOp::create(
        b, name, b.getStringAttr("private"), globalMemrefType, denseAttr,
        /*constant=*/true, /*alignment=*/IntegerAttr());
  }
  return memref::GetGlobalOp::create(b, globalOp.getType(),
                                     globalOp.getSymName());
}

template <unsigned W>
void rewriteNTT(NTTOp op, PatternRewriter& rewriter, ParamsAttr paramsAttr,
                int64_t lStart, int64_t lCount) {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto module = op->getParentOfType<ModuleOp>();
  Type intType = IntegerType::get(b.getContext(), W);

  auto params = paramsFromAttr<W>(paramsAttr);
  auto tables = cheddar_ref::Tables<W>::Generate(
      params, /*max_limbs=*/params.all_primes.size());
  size_t degree = params.degree;

  Value c0 = arith::ConstantIndexOp::create(b, 0);
  Value c1 = arith::ConstantIndexOp::create(b, 1);
  Value c2 = arith::ConstantIndexOp::create(b, 2);
  Value cDegree = arith::ConstantIndexOp::create(b, degree);
  Value cHalfDegree = arith::ConstantIndexOp::create(b, degree / 2);
  Value cLogN = arith::ConstantIndexOp::create(b, params.log_n);

  for (int64_t l = lStart; l < lStart + lCount; ++l) {
    Value lVal = arith::ConstantIndexOp::create(b, l);

    bool isSkipped = false;
    if (op.getSkipRange() && op.getSkipRange()->size() == 2) {
      int64_t skipStart = (*op.getSkipRange())[0];
      int64_t skipEnd = (*op.getSkipRange())[1];
      if (l >= skipStart && l < skipEnd) {
        isSkipped = true;
      }
    }

    if (isSkipped) {
      if (op.getSkipCopySrc()) {
        int64_t skipStart = (*op.getSkipRange())[0];
        int64_t skipInOffset = op.getSkipCopyInOffset();
        int64_t srcLimb = (l - skipStart) + skipInOffset;
        Value srcLimbVal = arith::ConstantIndexOp::create(b, srcLimb);
        scf::ForOp::create(
            b, c0, cDegree, c1, ValueRange{},
            [&](OpBuilder& nestedBuilder, Location loc, Value coeffIdx,
                ValueRange /*iterArgs*/) {
              ImplicitLocOpBuilder nb(loc, nestedBuilder);
              Value val = memref::LoadOp::create(
                  nb, op.getSkipCopySrc(), ValueRange{srcLimbVal, coeffIdx});
              memref::StoreOp::create(nb, val, op.getOut(),
                                      ValueRange{lVal, coeffIdx});
              scf::YieldOp::create(nb);
            });
      } else if (op.getOut() != op.getIn()) {
        scf::ForOp::create(b, c0, cDegree, c1, ValueRange{},
                           [&](OpBuilder& nestedBuilder, Location loc,
                               Value coeffIdx, ValueRange /*iterArgs*/) {
                             ImplicitLocOpBuilder nb(loc, nestedBuilder);
                             Value val = memref::LoadOp::create(
                                 nb, op.getIn(), ValueRange{lVal, coeffIdx});
                             memref::StoreOp::create(
                                 nb, val, op.getOut(),
                                 ValueRange{lVal, coeffIdx});
                             scf::YieldOp::create(nb);
                           });
      }
      continue;
    }

    int64_t primeTableIdx = l;
    if (op.getPrimeIndices() &&
        l < static_cast<int64_t>(op.getPrimeIndices()->size())) {
      primeTableIdx = (*op.getPrimeIndices())[l];
    } else if (op.getPrimeRange() && op.getPrimeRange()->size() == 2) {
      primeTableIdx = (*op.getPrimeRange())[0] + (l - lStart);
    } else if (op.getSkipRange() && op.getSkipRange()->size() == 2) {
      int64_t skipStart = (*op.getSkipRange())[0];
      int64_t skipEnd = (*op.getSkipRange())[1];
      if (skipStart == 0 && skipEnd > 0 && !params.aux_primes.empty()) {
        primeTableIdx =
            params.all_primes.size() - params.aux_primes.size() + (l - skipEnd);
      }
    }

    uint64_t q = tables.primes[primeTableIdx];
    auto modAttr = IntegerAttr::get(intType, APInt(W, q));
    auto modType = mod_arith::ModArithType::get(b.getContext(), modAttr);

    // 1. Copy input to output, with optional Montgomery conversion
    if (op.getMontgomeryConversion()) {
      typename cheddar_ref::Tables<W>::Word r2 = tables.R2_mod_q[primeTableIdx];
      Value r2Val = arith::ConstantOp::create(
          b, intType, b.getIntegerAttr(intType, APInt(W, r2)));
      Value r2Mod = mod_arith::EncapsulateOp::create(b, modType, r2Val);

      scf::ForOp::create(
          b, c0, cDegree, c1, ValueRange{},
          [&](OpBuilder& nestedBuilder, Location loc, Value coeffIdx,
              ValueRange /*iterArgs*/) {
            ImplicitLocOpBuilder nb(loc, nestedBuilder);
            Value inVal = memref::LoadOp::create(nb, op.getIn(),
                                                 ValueRange{lVal, coeffIdx});
            Value inMod = mod_arith::EncapsulateOp::create(nb, modType, inVal);
            Value montMod =
                mod_arith::MontMulOp::create(nb, modType, inMod, r2Mod);
            Value res = mod_arith::LiftOp::create(
                nb, intType, montMod, mod_arith::LiftType::STANDARD);
            memref::StoreOp::create(nb, res, op.getOut(),
                                    ValueRange{lVal, coeffIdx});
            scf::YieldOp::create(nb);
          });
    } else {
      scf::ForOp::create(b, c0, cDegree, c1, ValueRange{},
                         [&](OpBuilder& nestedBuilder, Location loc,
                             Value coeffIdx, ValueRange /*iterArgs*/) {
                           ImplicitLocOpBuilder nb(loc, nestedBuilder);
                           Value inVal = memref::LoadOp::create(
                               nb, op.getIn(), ValueRange{lVal, coeffIdx});
                           memref::StoreOp::create(nb, inVal, op.getOut(),
                                                   ValueRange{lVal, coeffIdx});
                           scf::YieldOp::create(nb);
                         });
    }

    // 2. Twiddle global for forward NTT
    const auto* twiddleData =
        tables.psi_rev_mont.data() + primeTableIdx * degree;
    Value twiddleMemref = getOrCreateTwiddleGlobal(
        b, module, "__cheddar_twiddle_fwd", q, degree, intType, twiddleData);

    // 3. Radix-2 Cooley-Tukey Decimation-in-Time (CT-DIT) butterfly network
    scf::ForOp::create(
        b, c0, cLogN, c1, ValueRange{},
        [&](OpBuilder& bStage, Location locStage, Value stage,
            ValueRange /*iterArgsStage*/) {
          ImplicitLocOpBuilder nbStage(locStage, bStage);
          Value m = arith::ShLIOp::create(nbStage, c1, stage);
          Value t = arith::ShRUIOp::create(nbStage, cHalfDegree, stage);
          Value stride = arith::MulIOp::create(nbStage, t, c2);

          scf::ForOp::create(
              nbStage, c0, m, c1, ValueRange{},
              [&](OpBuilder& bGroup, Location locGroup, Value i,
                  ValueRange /*iterArgsGroup*/) {
                ImplicitLocOpBuilder nbGroup(locGroup, bGroup);
                Value wIdx = arith::AddIOp::create(nbGroup, m, i);
                Value w = memref::LoadOp::create(nbGroup, twiddleMemref,
                                                 ValueRange{wIdx});
                Value wMod =
                    mod_arith::EncapsulateOp::create(nbGroup, modType, w);
                Value groupBase = arith::MulIOp::create(nbGroup, i, stride);

                scf::ForOp::create(
                    nbGroup, c0, t, c1, ValueRange{},
                    [&](OpBuilder& bFly, Location locFly, Value j,
                        ValueRange /*iterArgsFly*/) {
                      ImplicitLocOpBuilder nbFly(locFly, bFly);
                      Value idx0 = arith::AddIOp::create(nbFly, groupBase, j);
                      Value idx1 = arith::AddIOp::create(nbFly, idx0, t);

                      Value a = memref::LoadOp::create(nbFly, op.getOut(),
                                                       ValueRange{lVal, idx0});
                      Value bVal = memref::LoadOp::create(
                          nbFly, op.getOut(), ValueRange{lVal, idx1});

                      Value aMod =
                          mod_arith::EncapsulateOp::create(nbFly, modType, a);
                      Value bMod = mod_arith::EncapsulateOp::create(
                          nbFly, modType, bVal);
                      Value multMod = mod_arith::MontMulOp::create(
                          nbFly, modType, bMod, wMod);
                      Value aNewMod = mod_arith::AddOp::create(nbFly, modType,
                                                               aMod, multMod);
                      Value bNewMod = mod_arith::SubOp::create(nbFly, modType,
                                                               aMod, multMod);

                      Value aNew = mod_arith::LiftOp::create(
                          nbFly, intType, aNewMod,
                          mod_arith::LiftType::STANDARD);
                      Value bNew = mod_arith::LiftOp::create(
                          nbFly, intType, bNewMod,
                          mod_arith::LiftType::STANDARD);

                      memref::StoreOp::create(nbFly, aNew, op.getOut(),
                                              ValueRange{lVal, idx0});
                      memref::StoreOp::create(nbFly, bNew, op.getOut(),
                                              ValueRange{lVal, idx1});
                      scf::YieldOp::create(nbFly);
                    });
                scf::YieldOp::create(nbGroup);
              });
          scf::YieldOp::create(nbStage);
        });
  }
}

template <unsigned W>
void rewriteINTT(INTTOp op, PatternRewriter& rewriter, ParamsAttr paramsAttr,
                 int64_t inStart, int64_t outStart, int64_t lCount) {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto module = op->getParentOfType<ModuleOp>();
  Type intType = IntegerType::get(b.getContext(), W);

  auto params = paramsFromAttr<W>(paramsAttr);
  auto tables = cheddar_ref::Tables<W>::Generate(
      params, /*max_limbs=*/params.all_primes.size());
  size_t degree = params.degree;

  Value c0 = arith::ConstantIndexOp::create(b, 0);
  Value c1 = arith::ConstantIndexOp::create(b, 1);
  Value c2 = arith::ConstantIndexOp::create(b, 2);
  Value cDegree = arith::ConstantIndexOp::create(b, degree);
  Value cHalfDegree = arith::ConstantIndexOp::create(b, degree / 2);
  Value cLogN = arith::ConstantIndexOp::create(b, params.log_n);

  for (int64_t idx = 0; idx < lCount; ++idx) {
    int64_t inL = inStart + idx;
    if (op.getInSplit() && op.getInSplit()->size() == 2) {
      int64_t splitCount = (*op.getInSplit())[0];
      int64_t extraLimbs = (*op.getInSplit())[1];
      if (idx >= splitCount) inL += extraLimbs;
    }
    int64_t outL = outStart + idx;
    Value inLVal = arith::ConstantIndexOp::create(b, inL);
    Value outLVal = arith::ConstantIndexOp::create(b, outL);
    int64_t primeTableIdx = inL;
    if (op.getPrimeIndices() &&
        idx < static_cast<int64_t>(op.getPrimeIndices()->size())) {
      primeTableIdx = (*op.getPrimeIndices())[idx];
    } else if (op.getPrimeRange() && op.getPrimeRange()->size() == 2) {
      primeTableIdx = (*op.getPrimeRange())[0] + idx;
    } else if (!params.aux_primes.empty() && inStart > 0 &&
               inStart + lCount ==
                   cast<MemRefType>(op.getIn().getType()).getDimSize(0)) {
      primeTableIdx = params.all_primes.size() - params.aux_primes.size() + idx;
    }
    uint64_t q = tables.primes[primeTableIdx];
    auto modAttr = IntegerAttr::get(intType, APInt(W, q));
    auto modType = mod_arith::ModArithType::get(b.getContext(), modAttr);

    // 1. Copy input to output buffer
    scf::ForOp::create(b, c0, cDegree, c1, ValueRange{},
                       [&](OpBuilder& nestedBuilder, Location loc,
                           Value coeffIdx, ValueRange /*iterArgs*/) {
                         ImplicitLocOpBuilder nb(loc, nestedBuilder);
                         Value inVal = memref::LoadOp::create(
                             nb, op.getIn(), ValueRange{inLVal, coeffIdx});
                         memref::StoreOp::create(nb, inVal, op.getOut(),
                                                 ValueRange{outLVal, coeffIdx});
                         scf::YieldOp::create(nb);
                       });

    // 2. Twiddle global for inverse INTT
    const auto* twiddleData =
        tables.psi_inv_rev_mont.data() + primeTableIdx * degree;
    Value invTwiddleMemref = getOrCreateTwiddleGlobal(
        b, module, "__cheddar_twiddle_inv", q, degree, intType, twiddleData);

    // 3. Radix-2 Gentleman-Sande Decimation-in-Frequency (GS-DIF) butterfly
    scf::ForOp::create(
        b, c0, cLogN, c1, ValueRange{},
        [&](OpBuilder& bStage, Location locStage, Value stage,
            ValueRange /*iterArgsStage*/) {
          ImplicitLocOpBuilder nbStage(locStage, bStage);
          Value t = arith::ShLIOp::create(nbStage, c1, stage);
          Value m = arith::ShRUIOp::create(nbStage, cHalfDegree, stage);
          Value stride = arith::MulIOp::create(nbStage, t, c2);

          scf::ForOp::create(
              nbStage, c0, m, c1, ValueRange{},
              [&](OpBuilder& bGroup, Location locGroup, Value i,
                  ValueRange /*iterArgsGroup*/) {
                ImplicitLocOpBuilder nbGroup(locGroup, bGroup);
                Value wIdx = arith::AddIOp::create(nbGroup, m, i);
                Value w = memref::LoadOp::create(nbGroup, invTwiddleMemref,
                                                 ValueRange{wIdx});
                Value wMod =
                    mod_arith::EncapsulateOp::create(nbGroup, modType, w);
                Value groupBase = arith::MulIOp::create(nbGroup, i, stride);

                scf::ForOp::create(
                    nbGroup, c0, t, c1, ValueRange{},
                    [&](OpBuilder& bFly, Location locFly, Value j,
                        ValueRange /*iterArgsFly*/) {
                      ImplicitLocOpBuilder nbFly(locFly, bFly);
                      Value idx0 = arith::AddIOp::create(nbFly, groupBase, j);
                      Value idx1 = arith::AddIOp::create(nbFly, idx0, t);

                      Value a = memref::LoadOp::create(
                          nbFly, op.getOut(), ValueRange{outLVal, idx0});
                      Value bVal = memref::LoadOp::create(
                          nbFly, op.getOut(), ValueRange{outLVal, idx1});

                      Value aMod =
                          mod_arith::EncapsulateOp::create(nbFly, modType, a);
                      Value bMod = mod_arith::EncapsulateOp::create(
                          nbFly, modType, bVal);
                      Value aNewMod =
                          mod_arith::AddOp::create(nbFly, modType, aMod, bMod);
                      Value diffMod =
                          mod_arith::SubOp::create(nbFly, modType, aMod, bMod);
                      Value bNewMod = mod_arith::MontMulOp::create(
                          nbFly, modType, diffMod, wMod);

                      Value aNew = mod_arith::LiftOp::create(
                          nbFly, intType, aNewMod,
                          mod_arith::LiftType::STANDARD);
                      Value bNew = mod_arith::LiftOp::create(
                          nbFly, intType, bNewMod,
                          mod_arith::LiftType::STANDARD);

                      memref::StoreOp::create(nbFly, aNew, op.getOut(),
                                              ValueRange{outLVal, idx0});
                      memref::StoreOp::create(nbFly, bNew, op.getOut(),
                                              ValueRange{outLVal, idx1});
                      scf::YieldOp::create(nbFly);
                    });
                scf::YieldOp::create(nbGroup);
              });
          scf::YieldOp::create(nbStage);
        });

    // 4. Epilogue / Post-processing
    Value cVal;
    mod_arith::LiftType liftType = mod_arith::LiftType::STANDARD;

    if (op.getEpilogue() == INTTEpilogue::None) {
      typename cheddar_ref::Tables<W>::Word c =
          op.getMontgomeryConversion() ? tables.N_inv[primeTableIdx]
                                       : tables.N_inv_mont[primeTableIdx];
      cVal = arith::ConstantOp::create(b, intType,
                                       b.getIntegerAttr(intType, APInt(W, c)));
    } else if (op.getEpilogue() == INTTEpilogue::MultConst) {
      Value cstIdx = arith::ConstantIndexOp::create(b, idx);
      cVal = memref::LoadOp::create(b, op.getCst(), ValueRange{cstIdx});
    } else if (op.getEpilogue() == INTTEpilogue::MultConstNormalize) {
      Value cstIdx = arith::ConstantIndexOp::create(b, idx);
      cVal = memref::LoadOp::create(b, op.getCst(), ValueRange{cstIdx});
      liftType = mod_arith::LiftType::CENTERED;
    }

    Value cMod = mod_arith::EncapsulateOp::create(b, modType, cVal);

    scf::ForOp::create(
        b, c0, cDegree, c1, ValueRange{},
        [&](OpBuilder& nestedBuilder, Location loc, Value coeffIdx,
            ValueRange /*iterArgs*/) {
          ImplicitLocOpBuilder nb(loc, nestedBuilder);
          Value x = memref::LoadOp::create(nb, op.getOut(),
                                           ValueRange{outLVal, coeffIdx});
          Value xMod = mod_arith::EncapsulateOp::create(nb, modType, x);
          Value prodMod = mod_arith::MontMulOp::create(nb, modType, xMod, cMod);
          Value res = mod_arith::LiftOp::create(nb, intType, prodMod, liftType);
          memref::StoreOp::create(nb, res, op.getOut(),
                                  ValueRange{outLVal, coeffIdx});
          scf::YieldOp::create(nb);
        });
  }
}

struct ConvertNTTOp : public OpRewritePattern<NTTOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NTTOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent ModuleOp");
    auto params = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!params) return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto [lStart, lCount] =
        getLimbRange(op.getOutLimbRange(), outType.getDimSize(0));

    if (params.getWord() == 32) {
      rewriteNTT<32>(op, rewriter, params, lStart, lCount);
    } else if (params.getWord() == 64) {
      rewriteNTT<64>(op, rewriter, params, lStart, lCount);
    } else {
      return op.emitOpError("unsupported word size: ") << params.getWord();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertINTTOp : public OpRewritePattern<INTTOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(INTTOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent ModuleOp");
    auto params = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!params) return op.emitOpError("missing cheddar_gpu.params on module");

    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t inStart = 0;
    int64_t outStart = 0;
    int64_t count = outType.getDimSize(0);
    if (op.getInLimbRange() && op.getOutLimbRange()) {
      auto inRange = *op.getInLimbRange();
      auto outRange = *op.getOutLimbRange();
      inStart = inRange[0];
      outStart = outRange[0];
      count = outRange[1];
    } else if (op.getOutLimbRange()) {
      auto outRange = *op.getOutLimbRange();
      outStart = outRange[0];
      inStart = outRange[0];
      count = outRange[1];
    } else if (op.getInLimbRange()) {
      auto inRange = *op.getInLimbRange();
      inStart = inRange[0];
      outStart = inRange[0];
      count = inRange[1];
    }

    if (params.getWord() == 32) {
      rewriteINTT<32>(op, rewriter, params, inStart, outStart, count);
    } else if (params.getWord() == 64) {
      rewriteINTT<64>(op, rewriter, params, inStart, outStart, count);
    } else {
      return op.emitOpError("unsupported word size: ") << params.getWord();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateNTTToSCFPatterns(RewritePatternSet& patterns) {
  patterns.add<ConvertNTTOp, ConvertINTTOp>(patterns.getContext());
}

void configureNTTConversionTarget(ConversionTarget& target) {
  target.addIllegalOp<NTTOp, INTTOp>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
