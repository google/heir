#include "lib/Dialect/CheddarGPU/Transforms/ExpandModSwitch/ExpandModSwitch.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Utils/CheddarRef/ModSwitchTables.h"
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
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

#define GEN_PASS_DEF_EXPANDMODSWITCH
#include "lib/Dialect/CheddarGPU/Transforms/ExpandModSwitch/ExpandModSwitch.h.inc"

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
    globalOp = memref::GlobalOp::create(
        b, name, b.getStringAttr("private"), globalMemrefType, denseAttr,
        /*constant=*/true, /*alignment=*/IntegerAttr());
  }
  return memref::GetGlobalOp::create(b, globalOp.getType(),
                                     globalOp.getSymName());
}

template <typename Word>
Value getOrCreate2DGlobal(ImplicitLocOpBuilder& b, ModuleOp module,
                          StringRef name, int64_t dim0, int64_t dim1,
                          ArrayRef<Word> data, Type intType) {
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
    auto tensorType = RankedTensorType::get({dim0, dim1}, intType);
    auto denseAttr = DenseElementsAttr::get(tensorType, elements);
    auto globalMemrefType = MemRefType::get({dim0, dim1}, intType);
    globalOp = memref::GlobalOp::create(
        b, name, b.getStringAttr("private"), globalMemrefType, denseAttr,
        /*constant=*/true, /*alignment=*/IntegerAttr());
  }
  return memref::GetGlobalOp::create(b, globalOp.getType(),
                                     globalOp.getSymName());
}

// === Expand PseudoModUpOp ===

struct ExpandPseudoModUpOp : public OpRewritePattern<PseudoModUpOp> {
  using OpRewritePattern<PseudoModUpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PseudoModUpOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent module");
    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!paramsAttr)
      return op.emitOpError("missing cheddar_gpu.params attribute");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    uint32_t W = paramsAttr.getWord();
    Type intType = IntegerType::get(b.getContext(), W);
    int level = op.getLevel();

    std::string name =
        "__cheddar_pseudo_mod_up_p_prod_l" + std::to_string(level);

    int primeOffset = 0;
    Value pProdMemref;
    if (W == 32) {
      auto params = paramsFromAttr<32>(paramsAttr);
      auto tbl =
          cheddar_ref::ModSwitchTables32::GeneratePseudoModUp(params, level);
      pProdMemref =
          getOrCreate1DGlobal<uint32_t>(b, module, name, tbl.p_prod, intType);
      int num_ter = (level == -1) ? (params.level_config.at(0).second -
                                     params.additional_base.second)
                                  : params.level_config.at(level).second;
      primeOffset = static_cast<int>(params.ter_primes.size()) - num_ter;
    } else {
      auto params = paramsFromAttr<64>(paramsAttr);
      auto tbl =
          cheddar_ref::ModSwitchTables64::GeneratePseudoModUp(params, level);
      pProdMemref =
          getOrCreate1DGlobal<uint64_t>(b, module, name, tbl.p_prod, intType);
    }

    IntegerAttr primeOffsetAttr =
        primeOffset > 0 ? b.getI64IntegerAttr(primeOffset) : IntegerAttr{};
    rewriter.replaceOpWithNewOp<MulConstOp>(
        op, op.getOut(), op.getIn(), pProdMemref, op.getInForm(),
        op.getOutForm(), /*in_limb_range=*/nullptr,
        /*out_limb_range=*/nullptr, primeOffsetAttr);
    return success();
  }
};

// === Expand ModUpOp ===

struct ExpandModUpOp : public OpRewritePattern<ModUpOp> {
  using OpRewritePattern<ModUpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModUpOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent module");
    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!paramsAttr)
      return op.emitOpError("missing cheddar_gpu.params attribute");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    uint32_t W = paramsAttr.getWord();
    Type intType = IntegerType::get(b.getContext(), W);
    int level = op.getLevel();

    auto inType = cast<MemRefType>(op.getIn().getType());
    int64_t numQPrimes = inType.getDimSize(0);
    int64_t degree = inType.getDimSize(1);

    // 1. INTT with mult_const_normalize
    std::string modUp1Name = "__cheddar_mod_up1_l" + std::to_string(level);
    Value modUp1Memref;
    int beta = 0;
    struct DigitInfo {
      int prime_index_start;
      int prime_index_end;
      int src_len;
      int dst_len;
      Value tableMemref;
    };
    SmallVector<DigitInfo> digits;

    if (W == 32) {
      auto params = paramsFromAttr<32>(paramsAttr);
      auto tbl = cheddar_ref::ModSwitchTables32::GenerateModUp(params, level);
      modUp1Memref = getOrCreate1DGlobal<uint32_t>(b, module, modUp1Name,
                                                   tbl.mod_up1, intType);
      beta = tbl.beta;
      for (int i = 0; i < beta; ++i) {
        std::string tblName = "__cheddar_mod_up2_l" + std::to_string(level) +
                              "_digit" + std::to_string(i);
        Value tMem = getOrCreate2DGlobal<int32_t>(
            b, module, tblName, tbl.digits[i].dst_len, tbl.digits[i].src_len,
            tbl.digits[i].table, intType);
        digits.push_back({tbl.digits[i].prime_index_start,
                          tbl.digits[i].prime_index_end, tbl.digits[i].src_len,
                          tbl.digits[i].dst_len, tMem});
      }
    } else {
      auto params = paramsFromAttr<64>(paramsAttr);
      auto tbl = cheddar_ref::ModSwitchTables64::GenerateModUp(params, level);
      modUp1Memref = getOrCreate1DGlobal<uint64_t>(b, module, modUp1Name,
                                                   tbl.mod_up1, intType);
      beta = tbl.beta;
      for (int i = 0; i < beta; ++i) {
        std::string tblName = "__cheddar_mod_up2_l" + std::to_string(level) +
                              "_digit" + std::to_string(i);
        Value tMem = getOrCreate2DGlobal<int64_t>(
            b, module, tblName, tbl.digits[i].dst_len, tbl.digits[i].src_len,
            tbl.digits[i].table, intType);
        digits.push_back({tbl.digits[i].prime_index_start,
                          tbl.digits[i].prime_index_end, tbl.digits[i].src_len,
                          tbl.digits[i].dst_len, tMem});
      }
    }

    auto srcInttType = MemRefType::get({numQPrimes, degree}, intType);
    Value srcIntt = memref::AllocOp::create(b, srcInttType);

    // INTT(mult_const_normalize)
    INTTEpilogueAttr epilogueAttr =
        INTTEpilogueAttr::get(b.getContext(), INTTEpilogue::MultConstNormalize);
    b.create<INTTOp>(srcIntt, op.getIn(), modUp1Memref,
                     FormAttr::get(b.getContext(), Form::Eval),
                     FormAttr::get(b.getContext(), Form::CoeffCentered),
                     /*montgomery_conversion=*/b.getBoolAttr(true),
                     epilogueAttr,
                     /*in_limb_range=*/DenseI64ArrayAttr(),
                     /*out_limb_range=*/DenseI64ArrayAttr(),
                     /*prime_range=*/DenseI64ArrayAttr());

    if (static_cast<int>(op.getOuts().size()) != beta) {
      return op.emitOpError("mod_up requires ")
             << beta << " outputs for beta = " << beta << ", but got "
             << op.getOuts().size();
    }

    int64_t totalLimbs =
        cast<MemRefType>(op.getOuts()[0].getType()).getDimSize(0);
    int num_ter = 0;
    int max_ter = 0;
    if (W == 32) {
      auto params = paramsFromAttr<32>(paramsAttr);
      max_ter = static_cast<int>(params.ter_primes.size());
      num_ter = (level == -1) ? (params.level_config.at(0).second -
                                 params.additional_base.second)
                              : params.level_config.at(level).second;
    }
    int64_t qPrimeStart = max_ter - num_ter;
    int64_t auxPrimeStart =
        (W == 32) ? (paramsAttr.getTer().size() + paramsAttr.getMain().size())
                  : paramsAttr.getMain().size();

    SmallVector<int64_t> dstPrimeIndices;
    for (int64_t l = 0; l < totalLimbs; ++l) {
      if (l < numQPrimes) {
        dstPrimeIndices.push_back(qPrimeStart + l);
      } else {
        dstPrimeIndices.push_back(auxPrimeStart + (l - numQPrimes));
      }
    }

    // 2. Process each digit i
    for (int i = 0; i < beta; ++i) {
      Value dst_i = op.getOuts()[i];
      const auto& digit = digits[i];

      // BConv for destination limbs outside [prime_index_start,
      // prime_index_end)
      DenseI64ArrayAttr skipRangeAttr = b.getDenseI64ArrayAttr(
          {digit.prime_index_start, digit.prime_index_end});
      DenseI64ArrayAttr inLimbRangeAttr =
          b.getDenseI64ArrayAttr({digit.prime_index_start, digit.src_len});

      SmallVector<int64_t> bconvDstPrimes;
      for (int64_t l = 0; l < totalLimbs; ++l) {
        if (l >= digit.prime_index_start && l < digit.prime_index_end) continue;
        int64_t pIdx = dstPrimeIndices[l];
        int64_t primeVal = 0;
        if (W == 32) {
          auto params = paramsFromAttr<32>(paramsAttr);
          primeVal = static_cast<int64_t>(params.all_primes[pIdx]);
        } else {
          auto params = paramsFromAttr<64>(paramsAttr);
          primeVal = static_cast<int64_t>(params.all_primes[pIdx]);
        }
        bconvDstPrimes.push_back(primeVal);
      }

      b.create<BConvOp>(dst_i, srcIntt, digit.tableMemref,
                        FormAttr::get(b.getContext(), Form::CoeffCentered),
                        FormAttr::get(b.getContext(), Form::Coeff),
                        skipRangeAttr, inLimbRangeAttr,
                        /*out_limb_range=*/nullptr,
                        b.getDenseI64ArrayAttr(bconvDstPrimes));

      // NTT on dst_i with skip_range and prime_indices, folding verbatim copy
      // from op.getIn() into Phase 1 of NTT.
      Value skipCopySrc = digit.src_len > 0 ? op.getIn() : Value();
      int64_t skipCopyInOffset =
          digit.src_len > 0 ? digit.prime_index_start : 0;
      b.create<NTTOp>(
          dst_i, dst_i, skipCopySrc, FormAttr::get(b.getContext(), Form::Coeff),
          FormAttr::get(b.getContext(), Form::Eval),
          /*montgomery_conversion=*/b.getBoolAttr(false), skipRangeAttr,
          /*in_limb_range=*/DenseI64ArrayAttr(),
          /*out_limb_range=*/DenseI64ArrayAttr(),
          /*prime_range=*/DenseI64ArrayAttr(),
          b.getDenseI64ArrayAttr(dstPrimeIndices),
          b.getI64IntegerAttr(skipCopyInOffset));
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// === Expand ModDownOp ===

struct ExpandModDownOp : public OpRewritePattern<ModDownOp> {
  using OpRewritePattern<ModDownOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModDownOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent module");
    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!paramsAttr)
      return op.emitOpError("missing cheddar_gpu.params attribute");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    uint32_t W = paramsAttr.getWord();
    Type intType = IntegerType::get(b.getContext(), W);
    int level = op.getLevel();

    auto outType = cast<MemRefType>(op.getOut().getType());
    auto inType = cast<MemRefType>(op.getIn().getType());
    int64_t numQPrimes = outType.getDimSize(0);
    int64_t totalInLimbs = inType.getDimSize(0);
    int64_t numAux = totalInLimbs - numQPrimes;
    int64_t degree = outType.getDimSize(1);

    std::string modDown1Name = "__cheddar_mod_down1_l" + std::to_string(level);
    std::string modDown2Name = "__cheddar_mod_down2_l" + std::to_string(level);
    std::string invPProdName =
        "__cheddar_mod_down_inv_p_prod_l" + std::to_string(level);

    Value modDown1Memref;
    Value modDown2Memref;
    Value invPProdMemref;

    if (W == 32) {
      auto params = paramsFromAttr<32>(paramsAttr);
      auto tbl = cheddar_ref::ModSwitchTables32::GenerateModDown(params, level);
      modDown1Memref = getOrCreate1DGlobal<uint32_t>(b, module, modDown1Name,
                                                     tbl.mod_down1, intType);
      modDown2Memref =
          getOrCreate2DGlobal<int32_t>(b, module, modDown2Name, tbl.dst_len,
                                       tbl.src_len, tbl.mod_down2, intType);
      invPProdMemref = getOrCreate1DGlobal<uint32_t>(
          b, module, invPProdName, tbl.inv_prime_prod, intType);
    } else {
      auto params = paramsFromAttr<64>(paramsAttr);
      auto tbl = cheddar_ref::ModSwitchTables64::GenerateModDown(params, level);
      modDown1Memref = getOrCreate1DGlobal<uint64_t>(b, module, modDown1Name,
                                                     tbl.mod_down1, intType);
      modDown2Memref =
          getOrCreate2DGlobal<int64_t>(b, module, modDown2Name, tbl.dst_len,
                                       tbl.src_len, tbl.mod_down2, intType);
      invPProdMemref = getOrCreate1DGlobal<uint64_t>(
          b, module, invPProdName, tbl.inv_prime_prod, intType);
    }

    // 1. INTT on aux limbs of input
    auto auxInttType = MemRefType::get({totalInLimbs, degree}, intType);
    Value auxIntt = memref::AllocOp::create(b, auxInttType);

    DenseI64ArrayAttr inLimbRange =
        b.getDenseI64ArrayAttr({numQPrimes, numAux});
    DenseI64ArrayAttr outLimbRange =
        b.getDenseI64ArrayAttr({numQPrimes, numAux});
    int64_t auxPrimeStart =
        (W == 32) ? (paramsAttr.getTer().size() + paramsAttr.getMain().size())
                  : paramsAttr.getMain().size();
    DenseI64ArrayAttr primeRange =
        b.getDenseI64ArrayAttr({auxPrimeStart, numAux});
    INTTEpilogueAttr epilogueAttr =
        INTTEpilogueAttr::get(b.getContext(), INTTEpilogue::MultConstNormalize);
    b.create<INTTOp>(auxIntt, op.getIn(), modDown1Memref,
                     FormAttr::get(b.getContext(), Form::Eval),
                     FormAttr::get(b.getContext(), Form::CoeffCentered),
                     /*montgomery_conversion=*/b.getBoolAttr(true),
                     epilogueAttr, inLimbRange, outLimbRange, primeRange);

    // 2. BConv from auxIntt to temp buffer of shape [numQPrimes, degree]
    auto bconvOutType = MemRefType::get({numQPrimes, degree}, intType);
    Value bconvOut = memref::AllocOp::create(b, bconvOutType);

    SmallVector<int64_t> bconvDstPrimes;
    SmallVector<int64_t> bconvDstPrimeIndices;
    int64_t qPrimeStart = 0;
    if (W == 32) {
      auto params = paramsFromAttr<32>(paramsAttr);
      int max_ter = static_cast<int>(params.ter_primes.size());
      int num_ter = (level == -1) ? (params.level_config.at(0).second -
                                     params.additional_base.second)
                                  : params.level_config.at(level).second;
      qPrimeStart = max_ter - num_ter;
      for (int64_t l = 0; l < numQPrimes; ++l) {
        bconvDstPrimes.push_back(
            static_cast<int64_t>(params.all_primes[qPrimeStart + l]));
        bconvDstPrimeIndices.push_back(qPrimeStart + l);
      }
    } else {
      auto params = paramsFromAttr<64>(paramsAttr);
      for (int64_t l = 0; l < numQPrimes; ++l) {
        bconvDstPrimes.push_back(static_cast<int64_t>(params.all_primes[l]));
        bconvDstPrimeIndices.push_back(l);
      }
    }

    b.create<BConvOp>(bconvOut, auxIntt, modDown2Memref,
                      FormAttr::get(b.getContext(), Form::CoeffCentered),
                      FormAttr::get(b.getContext(), Form::Coeff),
                      /*skip_range=*/nullptr, inLimbRange,
                      /*out_limb_range=*/nullptr,
                      b.getDenseI64ArrayAttr(bconvDstPrimes));

    // 3. Fused NTTModDownOp
    b.create<NTTModDownOp>(op.getOut(), bconvOut, invPProdMemref, op.getIn(),
                           /*src2_padding=*/Value(), /*pad_start=*/0,
                           /*pad_end=*/numQPrimes, Form::Coeff, Form::Eval,
                           /*src2_offset=*/0, /*prime_offset=*/0,
                           b.getDenseI64ArrayAttr(bconvDstPrimeIndices));

    rewriter.eraseOp(op);
    return success();
  }
};

// === Expand RescaleOp ===

struct ExpandRescaleOp : public OpRewritePattern<RescaleOp> {
  using OpRewritePattern<RescaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RescaleOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent module");
    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!paramsAttr)
      return op.emitOpError("missing cheddar_gpu.params attribute");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    uint32_t W = paramsAttr.getWord();
    Type intType = IntegerType::get(b.getContext(), W);
    int level = op.getLevel();

    auto inType = cast<MemRefType>(op.getIn().getType());
    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t degree = inType.getDimSize(1);
    int64_t nextNumQ = outType.getDimSize(0);

    std::string r1Name = "__cheddar_rescale1_l" + std::to_string(level);
    std::string r2Name = "__cheddar_rescale2_l" + std::to_string(level);
    std::string invPProdName =
        "__cheddar_rescale_inv_p_prod_l" + std::to_string(level);
    std::string paddingName =
        "__cheddar_rescale_padding_l" + std::to_string(level);

    Value rescale1Memref;
    Value rescale2Memref;
    Value invPProdMemref;
    Value paddingMemref;

    int src_len = 0;
    int dst_len = 0;
    int pad_start = 0;
    int pad_end = 0;

    int64_t inttSrcOffset = 0;
    int64_t src2Offset = 0;

    if (W == 32) {
      auto params = paramsFromAttr<32>(paramsAttr);
      auto tbl = cheddar_ref::ModSwitchTables32::GenerateRescale(params, level);
      rescale1Memref = getOrCreate1DGlobal<uint32_t>(b, module, r1Name,
                                                     tbl.rescale1, intType);
      rescale2Memref = getOrCreate2DGlobal<int32_t>(
          b, module, r2Name, tbl.dst_len, tbl.src_len, tbl.rescale2, intType);
      invPProdMemref = getOrCreate1DGlobal<uint32_t>(
          b, module, invPProdName, tbl.inv_prime_prod, intType);
      paddingMemref = getOrCreate1DGlobal<uint32_t>(b, module, paddingName,
                                                    tbl.src2_padding, intType);
      src_len = tbl.src_len;
      dst_len = tbl.dst_len;
      pad_start = tbl.pad_start;
      pad_end = tbl.pad_end;

      int num_ter = params.level_config.at(level).second;
      int next_ter = params.level_config.at(level - 1).second;
      int num_main = params.level_config.at(level).first;
      int next_main = params.level_config.at(level - 1).first;
      int ter_diff = num_ter - next_ter;
      int main_diff = num_main - next_main;
      if (ter_diff > 0) {
        inttSrcOffset = 0;
      } else {
        inttSrcOffset = num_ter + num_main - main_diff;
      }
      src2Offset = std::max(0, num_ter - next_ter);
    } else {
      auto params = paramsFromAttr<64>(paramsAttr);
      auto tbl = cheddar_ref::ModSwitchTables64::GenerateRescale(params, level);
      rescale1Memref = getOrCreate1DGlobal<uint64_t>(b, module, r1Name,
                                                     tbl.rescale1, intType);
      rescale2Memref = getOrCreate2DGlobal<int64_t>(
          b, module, r2Name, tbl.dst_len, tbl.src_len, tbl.rescale2, intType);
      invPProdMemref = getOrCreate1DGlobal<uint64_t>(
          b, module, invPProdName, tbl.inv_prime_prod, intType);
      paddingMemref = getOrCreate1DGlobal<uint64_t>(b, module, paddingName,
                                                    tbl.src2_padding, intType);
      src_len = tbl.src_len;
      dst_len = tbl.dst_len;
      pad_start = tbl.pad_start;
      pad_end = tbl.pad_end;

      int num_ter = params.level_config.at(level).second;
      int next_ter = params.level_config.at(level - 1).second;
      int num_main = params.level_config.at(level).first;
      int next_main = params.level_config.at(level - 1).first;
      int ter_diff = num_ter - next_ter;
      int main_diff = num_main - next_main;
      if (ter_diff > 0) {
        inttSrcOffset = 0;
      } else {
        inttSrcOffset = num_ter + num_main - main_diff;
      }
      src2Offset = std::max(0, num_ter - next_ter);
    }

    // 1. INTT on rescale_src_primes
    auto srcInttType = MemRefType::get({inType.getDimSize(0), degree}, intType);
    Value srcIntt = memref::AllocOp::create(b, srcInttType);

    DenseI64ArrayAttr inLimbRange =
        b.getDenseI64ArrayAttr({inttSrcOffset, src_len});
    DenseI64ArrayAttr outLimbRange =
        b.getDenseI64ArrayAttr({inttSrcOffset, src_len});
    int64_t primeOffsetInAllPrimes = 0;
    if (W == 32) {
      auto params = paramsFromAttr<32>(paramsAttr);
      int num_ter = params.level_config.at(level).second;
      int max_ter = params.ter_primes.size();
      primeOffsetInAllPrimes = max_ter - num_ter;
    }
    int64_t rescalePrimeStart = primeOffsetInAllPrimes + inttSrcOffset;
    DenseI64ArrayAttr primeRange =
        b.getDenseI64ArrayAttr({rescalePrimeStart, src_len});
    INTTEpilogueAttr epilogueAttr =
        INTTEpilogueAttr::get(b.getContext(), INTTEpilogue::MultConstNormalize);
    b.create<INTTOp>(srcIntt, op.getIn(), rescale1Memref,
                     FormAttr::get(b.getContext(), Form::Eval),
                     FormAttr::get(b.getContext(), Form::CoeffCentered),
                     /*montgomery_conversion=*/b.getBoolAttr(true),
                     epilogueAttr, inLimbRange, outLimbRange, primeRange);

    // 2. BConv from srcIntt to next_level_primes
    auto bconvOutType = MemRefType::get({nextNumQ, degree}, intType);
    Value bconvOut = memref::AllocOp::create(b, bconvOutType);

    SmallVector<int64_t> bconvDstPrimes;
    SmallVector<int64_t> bconvDstPrimeIndices;
    if (W == 32) {
      auto params = paramsFromAttr<32>(paramsAttr);
      int max_ter = static_cast<int>(params.ter_primes.size());
      int next_ter = params.level_config.at(level - 1).second;
      int64_t nextQPrimeStart = max_ter - next_ter;
      for (int64_t l = 0; l < nextNumQ; ++l) {
        bconvDstPrimes.push_back(
            static_cast<int64_t>(params.all_primes[nextQPrimeStart + l]));
        bconvDstPrimeIndices.push_back(nextQPrimeStart + l);
      }
    } else {
      auto params = paramsFromAttr<64>(paramsAttr);
      for (int64_t l = 0; l < nextNumQ; ++l) {
        bconvDstPrimes.push_back(static_cast<int64_t>(params.all_primes[l]));
        bconvDstPrimeIndices.push_back(l);
      }
    }

    b.create<BConvOp>(bconvOut, srcIntt, rescale2Memref,
                      FormAttr::get(b.getContext(), Form::CoeffCentered),
                      FormAttr::get(b.getContext(), Form::Coeff),
                      /*skip_range=*/nullptr, inLimbRange,
                      /*out_limb_range=*/nullptr,
                      b.getDenseI64ArrayAttr(bconvDstPrimes));

    // 3. Fused NTTModDownOp
    b.create<NTTModDownOp>(op.getOut(), bconvOut, invPProdMemref, op.getIn(),
                           paddingMemref, /*pad_start=*/pad_start,
                           /*pad_end=*/pad_end, Form::Coeff, Form::Eval,
                           /*src2_offset=*/src2Offset, /*prime_offset=*/0,
                           b.getDenseI64ArrayAttr(bconvDstPrimeIndices));

    rewriter.eraseOp(op);
    return success();
  }
};

// === Expand ModDownRescaleOp ===

struct ExpandModDownRescaleOp : public OpRewritePattern<ModDownRescaleOp> {
  using OpRewritePattern<ModDownRescaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModDownRescaleOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent module");
    auto paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    if (!paramsAttr)
      return op.emitOpError("missing cheddar_gpu.params attribute");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    uint32_t W = paramsAttr.getWord();
    Type intType = IntegerType::get(b.getContext(), W);
    int level = op.getLevel();

    auto inType = cast<MemRefType>(op.getIn().getType());
    auto outType = cast<MemRefType>(op.getOut().getType());
    int64_t degree = inType.getDimSize(1);
    int64_t nextNumQ = outType.getDimSize(0);

    std::string mdr1Name =
        "__cheddar_mod_down_rescale1_l" + std::to_string(level);
    std::string mdr2Name =
        "__cheddar_mod_down_rescale2_l" + std::to_string(level);
    std::string invPProdName =
        "__cheddar_mod_down_rescale_inv_p_prod_l" + std::to_string(level);
    std::string paddingName =
        "__cheddar_mod_down_rescale_padding_l" + std::to_string(level);

    Value mdr1Memref;
    Value mdr2Memref;
    Value invPProdMemref;
    Value paddingMemref;

    int src_len = 0;
    int dst_len = 0;
    int pad_start = 0;
    int pad_end = 0;

    int64_t q_offset = 0;
    int64_t q_count = 0;
    int64_t p_offset = 0;
    int64_t p_count = 0;
    int64_t src2Offset = 0;

    if (W == 32) {
      auto params = paramsFromAttr<32>(paramsAttr);
      auto tbl =
          cheddar_ref::ModSwitchTables32::GenerateModDownRescale(params, level);
      src_len = tbl.src_len;
      dst_len = tbl.dst_len;
      pad_start = tbl.pad_start;
      pad_end = tbl.pad_end;

      int num_ter = params.level_config.at(level).second;
      int next_ter = params.level_config.at(level - 1).second;
      int num_main = params.level_config.at(level).first;
      int next_main = params.level_config.at(level - 1).first;
      int ter_diff = num_ter - next_ter;
      int main_diff = num_main - next_main;
      p_offset = num_ter + num_main;
      p_count = params.alpha;
      if (ter_diff > 0) {
        q_offset = 0;
        q_count = ter_diff;
      } else {
        q_offset = num_ter + num_main - main_diff;
        q_count = main_diff;
      }
      src2Offset = std::max(0, num_ter - next_ter);

      mdr1Memref = getOrCreate1DGlobal<uint32_t>(
          b, module, mdr1Name, tbl.mod_down_rescale1, intType);
      mdr2Memref = getOrCreate2DGlobal<int32_t>(b, module, mdr2Name,
                                                tbl.dst_len, tbl.src_len,
                                                tbl.mod_down_rescale2, intType);
      invPProdMemref = getOrCreate1DGlobal<uint32_t>(
          b, module, invPProdName, tbl.inv_prime_prod, intType);
      paddingMemref = getOrCreate1DGlobal<uint32_t>(b, module, paddingName,
                                                    tbl.src2_padding, intType);
    } else {
      auto params = paramsFromAttr<64>(paramsAttr);
      auto tbl =
          cheddar_ref::ModSwitchTables64::GenerateModDownRescale(params, level);
      src_len = tbl.src_len;
      dst_len = tbl.dst_len;
      pad_start = tbl.pad_start;
      pad_end = tbl.pad_end;

      int num_ter = params.level_config.at(level).second;
      int next_ter = params.level_config.at(level - 1).second;
      int num_main = params.level_config.at(level).first;
      int next_main = params.level_config.at(level - 1).first;
      int ter_diff = num_ter - next_ter;
      int main_diff = num_main - next_main;
      p_offset = num_ter + num_main;
      p_count = params.alpha;
      if (ter_diff > 0) {
        q_offset = 0;
        q_count = ter_diff;
      } else {
        q_offset = num_ter + num_main - main_diff;
        q_count = main_diff;
      }
      src2Offset = std::max(0, num_ter - next_ter);

      mdr1Memref = getOrCreate1DGlobal<uint64_t>(
          b, module, mdr1Name, tbl.mod_down_rescale1, intType);
      mdr2Memref = getOrCreate2DGlobal<int64_t>(b, module, mdr2Name,
                                                tbl.dst_len, tbl.src_len,
                                                tbl.mod_down_rescale2, intType);
      invPProdMemref = getOrCreate1DGlobal<uint64_t>(
          b, module, invPProdName, tbl.inv_prime_prod, intType);
      paddingMemref = getOrCreate1DGlobal<uint64_t>(b, module, paddingName,
                                                    tbl.src2_padding, intType);
    }

    // Allocate buffer for src_intt matching inType shape
    auto srcInttType = MemRefType::get({inType.getDimSize(0), degree}, intType);
    Value srcIntt = memref::AllocOp::create(b, srcInttType);

    // 1. INTT on Q and P limbs of rescale_src in a single launch
    INTTEpilogueAttr epilogueAttr =
        INTTEpilogueAttr::get(b.getContext(), INTTEpilogue::MultConstNormalize);
    int64_t qPrimeStart = 0;
    int64_t auxPrimeStart = 0;
    if (W == 32) {
      auto params = paramsFromAttr<32>(paramsAttr);
      int num_ter = params.level_config.at(level).second;
      int max_ter = params.ter_primes.size();
      qPrimeStart = (max_ter - num_ter) + q_offset;
      auxPrimeStart = max_ter + params.main_primes.size();
    } else {
      qPrimeStart = q_offset;
      auxPrimeStart = paramsAttr.getMain().size();
    }

    SmallVector<int64_t> inttPrimeIndices;
    for (int64_t i = 0; i < q_count; ++i) {
      inttPrimeIndices.push_back(qPrimeStart + i);
    }
    for (int64_t i = 0; i < p_count; ++i) {
      inttPrimeIndices.push_back(auxPrimeStart + i);
    }

    DenseI64ArrayAttr inSplitAttr;
    int64_t extra_limbs = p_offset - (q_offset + q_count);
    if (extra_limbs > 0) {
      inSplitAttr = b.getDenseI64ArrayAttr({q_count, extra_limbs});
    }

    DenseI64ArrayAttr inLimbRange = b.getDenseI64ArrayAttr({q_offset, src_len});
    DenseI64ArrayAttr outLimbRange = b.getDenseI64ArrayAttr({0, src_len});
    b.create<INTTOp>(srcIntt, op.getIn(), mdr1Memref,
                     FormAttr::get(b.getContext(), Form::Eval),
                     FormAttr::get(b.getContext(), Form::CoeffCentered),
                     /*montgomery_conversion=*/b.getBoolAttr(true),
                     epilogueAttr, inLimbRange, outLimbRange,
                     /*prime_range=*/DenseI64ArrayAttr(),
                     b.getDenseI64ArrayAttr(inttPrimeIndices), inSplitAttr);

    // 2. BConv from srcIntt to next_level_primes
    auto bconvOutType = MemRefType::get({nextNumQ, degree}, intType);
    Value bconvOut = memref::AllocOp::create(b, bconvOutType);

    SmallVector<int64_t> bconvDstPrimes;
    SmallVector<int64_t> bconvDstPrimeIndices;
    if (W == 32) {
      auto params = paramsFromAttr<32>(paramsAttr);
      int max_ter = static_cast<int>(params.ter_primes.size());
      int next_ter = params.level_config.at(level - 1).second;
      int64_t nextQPrimeStart = max_ter - next_ter;
      for (int64_t l = 0; l < nextNumQ; ++l) {
        bconvDstPrimes.push_back(
            static_cast<int64_t>(params.all_primes[nextQPrimeStart + l]));
        bconvDstPrimeIndices.push_back(nextQPrimeStart + l);
      }
    } else {
      auto params = paramsFromAttr<64>(paramsAttr);
      for (int64_t l = 0; l < nextNumQ; ++l) {
        bconvDstPrimes.push_back(static_cast<int64_t>(params.all_primes[l]));
        bconvDstPrimeIndices.push_back(l);
      }
    }

    b.create<BConvOp>(
        bconvOut, srcIntt, mdr2Memref,
        FormAttr::get(b.getContext(), Form::CoeffCentered),
        FormAttr::get(b.getContext(), Form::Coeff),
        /*skip_range=*/nullptr, b.getDenseI64ArrayAttr({0, src_len}),
        /*out_limb_range=*/nullptr, b.getDenseI64ArrayAttr(bconvDstPrimes));

    // 3. Fused NTTModDownOp
    b.create<NTTModDownOp>(op.getOut(), bconvOut, invPProdMemref, op.getIn(),
                           paddingMemref, /*pad_start=*/pad_start,
                           /*pad_end=*/pad_end, Form::Coeff, Form::Eval,
                           /*src2_offset=*/src2Offset, /*prime_offset=*/0,
                           b.getDenseI64ArrayAttr(bconvDstPrimeIndices));

    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandModSwitch : impl::ExpandModSwitchBase<ExpandModSwitch> {
  using ExpandModSwitchBase::ExpandModSwitchBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    ModuleOp module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<ExpandPseudoModUpOp, ExpandModUpOp, ExpandModDownOp,
                 ExpandRescaleOp, ExpandModDownRescaleOp>(context);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           scf::SCFDialect, CheddarGPUDialect>();
    target.addIllegalOp<PseudoModUpOp, ModUpOp, ModDownOp, RescaleOp,
                        ModDownRescaleOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createExpandModSwitchPass() {
  return std::make_unique<ExpandModSwitch>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
