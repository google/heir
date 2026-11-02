#include "lib/Dialect/CheddarGPU/Transforms/ExpandKeySwitch/ExpandKeySwitch.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Utils/CheddarRef/MontArith.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/Tables.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"    // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"      // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

#define GEN_PASS_DEF_EXPANDKEYSWITCH
#include "lib/Dialect/CheddarGPU/Transforms/ExpandKeySwitch/ExpandKeySwitch.h.inc"

namespace {

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
  p.L = p.ter_primes.size() + p.main_primes.size();

  for (auto attrPair : attr.getLevelConfig()) {
    auto arr = cast<ArrayAttr>(attrPair);
    if (!arr.empty() && isa<ArrayAttr>(arr[0])) {
      arr = cast<ArrayAttr>(arr[0]);
    }
    int m = cast<IntegerAttr>(arr[0]).getInt();
    int t = cast<IntegerAttr>(arr[1]).getInt();
    p.level_config.push_back({m, t});
  }

  return p;
}

template <typename Word>
Value getOrCreate1DGlobal(ImplicitLocOpBuilder& b, ModuleOp module,
                          StringRef name, ArrayRef<Word> data, Type intType) {
  auto globalOp = module.lookupSymbol<memref::GlobalOp>(name);
  if (!globalOp) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    SmallVector<APInt> elements;
    elements.reserve(data.size());
    unsigned width = intType.getIntOrFloatBitWidth();
    for (auto val : data) {
      elements.push_back(APInt(
          width,
          static_cast<uint64_t>(static_cast<std::make_unsigned_t<Word>>(val))));
    }
    auto tensorType =
        RankedTensorType::get({static_cast<int64_t>(data.size())}, intType);
    auto denseAttr = DenseElementsAttr::get(tensorType, elements);
    auto globalMemrefType =
        MemRefType::get({static_cast<int64_t>(data.size())}, intType);
    memref::GlobalOp::create(b, name, b.getStringAttr("private"),
                             globalMemrefType, denseAttr,
                             /*constant=*/true, /*alignment=*/nullptr);
  }
  auto globalMemrefType =
      MemRefType::get({static_cast<int64_t>(data.size())}, intType);
  return memref::GetGlobalOp::create(b, globalMemrefType, name);
}

template <unsigned W>
Value createPProdDevice(ImplicitLocOpBuilder& b, ModuleOp module, int level,
                        size_t num_q, size_t num_aux, size_t prime_offset,
                        const cheddar_ref::Params<W>& params, Type intType) {
  using Word = typename cheddar_ref::MontArithTraits<W>::Word;
  size_t total_limbs = num_q;
  std::vector<Word> p_prod_data(total_limbs, Word(0));

  for (size_t l = 0; l < num_q; ++l) {
    size_t prime_idx = l + prime_offset;
    Word q = params.all_primes[prime_idx];
    Word p_prod = 1;
    for (size_t a = 0; a < num_aux; ++a) {
      Word p_aux = params.aux_primes[a];
      p_prod = cheddar_ref::Tables<W>::MultMod(p_prod, p_aux, q);
    }
    p_prod_data[l] = cheddar_ref::MontArith<W>::ToMont(p_prod, q);
  }

  std::string levelStr =
      level < 0 ? "m" + std::to_string(-level) : std::to_string(level);
  std::string name = "__cheddar_pprod_l" + levelStr;
  return getOrCreate1DGlobal<Word>(b, module, name, p_prod_data, intType);
}

// === Expansion Pattern: KeySwitchNoModDownOp ===

struct ExpandKeySwitchNoModDownPattern
    : public OpRewritePattern<KeySwitchNoModDownOp> {
  using OpRewritePattern<KeySwitchNoModDownOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(KeySwitchNoModDownOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!paramsAttr) {
      return op.emitOpError("module missing 'cheddar_gpu.params' attribute");
    }

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    unsigned width = outType.getElementType().getIntOrFloatBitWidth();
    int level = op.getLevel();

    int64_t numQWithAux = outType.getDimSize(0);
    Type elemType = outType.getElementType();

    int64_t alpha = paramsAttr.getAlpha();
    int64_t numQ = 0;
    int64_t numAux = 0;
    size_t prime_offset = 0;
    size_t padded_num_q = 0;
    int beta = 0;

    if (level == -1) {
      numQ = 1;
      numAux = 1;  // sseNumAux = 1
      beta = 1;
      prime_offset = 0;
      padded_num_q = 1;
    } else {
      numAux = alpha;
      numQ = numQWithAux - alpha;
      size_t max_num_ter = paramsAttr.getTer().size();
      auto levelPair = cast<ArrayAttr>(paramsAttr.getLevelConfig()[level]);
      if (!levelPair.empty() && isa<ArrayAttr>(levelPair[0])) {
        levelPair = cast<ArrayAttr>(levelPair[0]);
      }
      int64_t numTer = cast<IntegerAttr>(levelPair[1]).getInt();
      prime_offset = max_num_ter - numTer;
      padded_num_q = numQ + prime_offset;
      beta = (padded_num_q + alpha - 1) / alpha;
    }

    // 1. ModUp on `in` producing beta buffers of shape [num_q + num_aux,
    // degree]
    SmallVector<Value> modUpOuts;
    modUpOuts.reserve(beta);
    for (int i = 0; i < beta; ++i) {
      modUpOuts.push_back(memref::AllocOp::create(b, outType));
    }

    ModUpOp::create(b, modUpOuts, op.getIn(), op.getLevel(), Form::Eval,
                    Form::Eval);

    // 2. Filter active digits
    SmallVector<Value> activeModUp;
    SmallVector<Value> activeEvkBx;
    SmallVector<Value> activeEvkAx;

    for (int i = 0; i < beta; ++i) {
      size_t prime_index_end =
          std::min((i + 1) * alpha, static_cast<int64_t>(padded_num_q));
      if (prime_index_end <= prime_offset) continue;
      activeModUp.push_back(modUpOuts[i]);
      activeEvkBx.push_back(op.getEvkBx()[i]);
      activeEvkAx.push_back(op.getEvkAx()[i]);
    }

    IntegerAttr primeOffsetAttr =
        prime_offset > 0 ? b.getI64IntegerAttr(prime_offset) : nullptr;
    IntegerAttr numAuxAttr = numAux > 0 ? b.getI64IntegerAttr(numAux) : nullptr;
    CPAccum2Op::create(b, op.getOutBx(), op.getOutAx(), activeEvkBx,
                       activeEvkAx, activeModUp, Form::Eval,
                       /*ct_forms=*/ArrayAttr(), /*common_forms=*/ArrayAttr(),
                       /*out_limb_range=*/DenseI64ArrayAttr(),
                       /*prime_offset=*/primeOffsetAttr,
                       /*num_aux=*/numAuxAttr);

    // 3. Add p_prod * in_bx and p_prod * in_ax if present
    if (op.getInBx() || op.getInAx()) {
      Value pProdTable;
      if (width == 32) {
        auto params = paramsFromAttr<32>(paramsAttr);
        pProdTable = createPProdDevice<32>(b, module, level, numQ, numAux,
                                           prime_offset, params, elemType);
      } else {
        auto params = paramsFromAttr<64>(paramsAttr);
        pProdTable = createPProdDevice<64>(b, module, level, numQ, numAux,
                                           prime_offset, params, elemType);
      }

      DenseI64ArrayAttr qLimbRange = b.getDenseI64ArrayAttr({0, numQ});
      auto evalFormAttr = FormAttr::get(b.getContext(), Form::Eval);
      if (op.getInBx() && op.getInAx()) {
        CPAccumAdd2Op::create(
            b, op.getOutBx(), op.getOutAx(), op.getOutBx(), op.getOutAx(),
            ValueRange{op.getInBx()}, ValueRange{op.getInAx()},
            ValueRange{pProdTable}, evalFormAttr, evalFormAttr,
            /*ct_forms=*/ArrayAttr(), /*common_forms=*/ArrayAttr(),
            /*out_limb_range=*/qLimbRange,
            /*prime_offset=*/nullptr,
            /*num_aux=*/nullptr);
      } else if (op.getInBx()) {
        CPAccumAddOp::create(
            b, op.getOutBx(), op.getOutBx(), ValueRange{op.getInBx()},
            ValueRange{pProdTable}, evalFormAttr, evalFormAttr,
            /*ct_forms=*/ArrayAttr(), /*common_forms=*/ArrayAttr(),
            /*out_limb_range=*/qLimbRange,
            /*prime_offset=*/nullptr,
            /*num_aux=*/nullptr);
      } else if (op.getInAx()) {
        CPAccumAddOp::create(
            b, op.getOutAx(), op.getOutAx(), ValueRange{op.getInAx()},
            ValueRange{pProdTable}, evalFormAttr, evalFormAttr,
            /*ct_forms=*/ArrayAttr(), /*common_forms=*/ArrayAttr(),
            /*out_limb_range=*/qLimbRange,
            /*prime_offset=*/nullptr,
            /*num_aux=*/nullptr);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// === Expansion Pattern: KeySwitchOp ===

struct ExpandKeySwitchPattern : public OpRewritePattern<KeySwitchOp> {
  using OpRewritePattern<KeySwitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(KeySwitchOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!paramsAttr) {
      return op.emitOpError("module missing 'cheddar_gpu.params' attribute");
    }

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    int64_t numQ = outType.getDimSize(0);
    int64_t degree = outType.getDimSize(1);
    int64_t alpha = paramsAttr.getAlpha();
    int64_t numAux = (op.getLevel() == -1) ? 1 : alpha;
    auto accumType =
        MemRefType::get({numQ + numAux, degree}, outType.getElementType());

    Value accBx = memref::AllocOp::create(b, accumType);
    Value accAx = memref::AllocOp::create(b, accumType);

    KeySwitchNoModDownOp::create(b, accBx, accAx, op.getInAx(), op.getInBx(),
                                 /*in_ax=*/Value(), op.getEvkBx(),
                                 op.getEvkAx(), op.getLevel(), Form::Eval,
                                 Form::Eval);

    ModDownOp::create(b, op.getOutBx(), accBx, op.getLevel(), Form::Eval,
                      Form::Eval);
    ModDownOp::create(b, op.getOutAx(), accAx, op.getLevel(), Form::Eval,
                      Form::Eval);

    rewriter.eraseOp(op);
    return success();
  }
};

// === Expansion Pattern: RelinearizeOp ===

struct ExpandRelinearizePattern : public OpRewritePattern<RelinearizeOp> {
  using OpRewritePattern<RelinearizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RelinearizeOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!paramsAttr) {
      return op.emitOpError("module missing 'cheddar_gpu.params' attribute");
    }

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    int64_t numQ = outType.getDimSize(0);
    int64_t degree = outType.getDimSize(1);
    int64_t alpha = paramsAttr.getAlpha();
    auto accumType =
        MemRefType::get({numQ + alpha, degree}, outType.getElementType());

    Value accBx = memref::AllocOp::create(b, accumType);
    Value accAx = memref::AllocOp::create(b, accumType);

    KeySwitchNoModDownOp::create(b, accBx, accAx, op.getInRx(), op.getInBx(),
                                 op.getInAx(), op.getEvkBx(), op.getEvkAx(),
                                 op.getLevel(), Form::Eval, Form::Eval);

    ModDownOp::create(b, op.getOutBx(), accBx, op.getLevel(), Form::Eval,
                      Form::Eval);
    ModDownOp::create(b, op.getOutAx(), accAx, op.getLevel(), Form::Eval,
                      Form::Eval);

    rewriter.eraseOp(op);
    return success();
  }
};

// === Expansion Pattern: RelinearizeRescaleOp ===

struct ExpandRelinearizeRescalePattern
    : public OpRewritePattern<RelinearizeRescaleOp> {
  using OpRewritePattern<RelinearizeRescaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RelinearizeRescaleOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!paramsAttr) {
      return op.emitOpError("module missing 'cheddar_gpu.params' attribute");
    }

    auto inType = cast<MemRefType>(op.getInBx().getType());
    int64_t numQLevel = inType.getDimSize(0);
    int64_t degree = inType.getDimSize(1);
    int64_t alpha = paramsAttr.getAlpha();
    auto accumType =
        MemRefType::get({numQLevel + alpha, degree}, inType.getElementType());

    Value accBx = memref::AllocOp::create(b, accumType);
    Value accAx = memref::AllocOp::create(b, accumType);

    KeySwitchNoModDownOp::create(b, accBx, accAx, op.getInRx(), op.getInBx(),
                                 op.getInAx(), op.getEvkBx(), op.getEvkAx(),
                                 op.getLevel(), Form::Eval, Form::Eval);

    ModDownRescaleOp::create(b, op.getOutBx(), accBx, op.getLevel(), Form::Eval,
                             Form::Eval);
    ModDownRescaleOp::create(b, op.getOutAx(), accAx, op.getLevel(), Form::Eval,
                             Form::Eval);

    rewriter.eraseOp(op);
    return success();
  }
};

// === Expansion Pattern: HRotOp ===

struct ExpandHRotPattern : public OpRewritePattern<HRotOp> {
  using OpRewritePattern<HRotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HRotOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!paramsAttr) {
      return op.emitOpError("module missing 'cheddar_gpu.params' attribute");
    }

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    int64_t degree = outType.getDimSize(1);

    Value tmpBx = memref::AllocOp::create(b, outType);
    Value tmpAx = memref::AllocOp::create(b, outType);

    KeySwitchOp::create(b, tmpBx, tmpAx, op.getInBx(), op.getInAx(),
                        op.getEvkBx(), op.getEvkAx(), op.getLevel(), Form::Eval,
                        Form::Eval);

    int64_t galoisFactor = 0;
    if (op.getGalois()) {
      galoisFactor = *op.getGalois();
    } else {
      // Compute 5^rot_dist mod 2N
      int64_t rotDist = op.getRotDist();
      int64_t mod = 2 * degree;
      int64_t base = 5;
      int64_t res = 1;
      int64_t exp = rotDist % (degree / 2);
      if (exp < 0) exp += (degree / 2);
      while (exp > 0) {
        if (exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
      }
      galoisFactor = res;
    }

    Permute2Op::create(b, op.getOutBx(), op.getOutAx(), tmpBx, tmpAx,
                       Form::Eval, Form::Eval,
                       b.getI64IntegerAttr(galoisFactor),
                       /*conj=*/false, DenseI64ArrayAttr(), DenseI64ArrayAttr(),
                       /*num_aux=*/nullptr);

    rewriter.eraseOp(op);
    return success();
  }
};

// === Expansion Pattern: HConjOp ===

struct ExpandHConjPattern : public OpRewritePattern<HConjOp> {
  using OpRewritePattern<HConjOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HConjOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    Value tmpBx = memref::AllocOp::create(b, outType);
    Value tmpAx = memref::AllocOp::create(b, outType);

    KeySwitchOp::create(b, tmpBx, tmpAx, op.getInBx(), op.getInAx(),
                        op.getEvkBx(), op.getEvkAx(), op.getLevel(), Form::Eval,
                        Form::Eval);

    Permute2Op::create(b, op.getOutBx(), op.getOutAx(), tmpBx, tmpAx,
                       Form::Eval, Form::Eval, IntegerAttr(), /*conj=*/true,
                       DenseI64ArrayAttr(), DenseI64ArrayAttr(),
                       /*num_aux=*/nullptr);

    rewriter.eraseOp(op);
    return success();
  }
};

// === Expansion Pattern: HRotAddOp ===

struct ExpandHRotAddPattern : public OpRewritePattern<HRotAddOp> {
  using OpRewritePattern<HRotAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HRotAddOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    auto outType = cast<MemRefType>(op.getOutBx().getType());
    int64_t degree = outType.getDimSize(1);

    int64_t galoisFactor = 0;
    if (op.getGalois().has_value()) {
      galoisFactor = *op.getGalois();
    } else {
      int64_t rotDist = op.getRotDist();
      int64_t mod = 2 * degree;
      int64_t base = 5;
      int64_t res = 1;
      int64_t exp = rotDist % (degree / 2);
      if (exp < 0) exp += (degree / 2);
      while (exp > 0) {
        if (exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
      }
      galoisFactor = res;
    }

    Value tmpBx = memref::AllocOp::create(b, outType);
    Value tmpAx = memref::AllocOp::create(b, outType);

    KeySwitchOp::create(b, tmpBx, tmpAx, op.getIn1Bx(), op.getIn1Ax(),
                        op.getEvkBx(), op.getEvkAx(), op.getLevel(), Form::Eval,
                        Form::Eval);

    DenseI64ArrayAttr galoisFactorsAttr =
        b.getDenseI64ArrayAttr({galoisFactor});

    PermuteAccum2Op::create(b, op.getOutBx(), op.getOutAx(),
                            /*acc_bx=*/op.getIn2Bx(), /*acc_ax=*/op.getIn2Ax(),
                            /*srcs_bx=*/ValueRange{tmpBx},
                            /*srcs_ax=*/ValueRange{tmpAx}, Form::Eval,
                            /*in_forms=*/ArrayAttr(), galoisFactorsAttr,
                            /*out_limb_range=*/DenseI64ArrayAttr(),
                            /*prime_offset=*/nullptr,
                            /*num_aux=*/nullptr);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandKeySwitchPass
    : public impl::ExpandKeySwitchBase<ExpandKeySwitchPass> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ExpandHRotAddPattern, ExpandHConjPattern, ExpandHRotPattern,
                 ExpandRelinearizeRescalePattern, ExpandRelinearizePattern,
                 ExpandKeySwitchPattern, ExpandKeySwitchNoModDownPattern>(
        context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createExpandKeySwitchPass() {
  return std::make_unique<ExpandKeySwitchPass>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
