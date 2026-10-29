#include "lib/Dialect/CheddarGPU/Transforms/ExpandHoist/ExpandHoist.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"
#include "lib/Utils/CheddarRef/BootConsts.h"
#include "lib/Utils/CheddarRef/Encoder.h"
#include "lib/Utils/CheddarRef/ModSwitchTables.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/Tables.h"
#include "llvm/include/llvm/ADT/APInt.h"                  // from @llvm-project
#include "llvm/include/llvm/ADT/ArrayRef.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"              // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"    // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"      // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_gpu {

#define GEN_PASS_DEF_EXPANDHOIST
#include "lib/Dialect/CheddarGPU/Transforms/ExpandHoist/ExpandHoist.h.inc"

namespace {

int64_t computeGaloisFactor(int64_t rotDist, int64_t degree) {
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
  return res;
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

  p.scale_info.log_default_scale = 40;
  p.scale_info.default_encryption_level = 13;
  p.scale_info.max_level =
      p.level_config.empty() ? 25 : static_cast<int>(p.level_config.size() - 1);

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
    auto memrefType =
        MemRefType::get({static_cast<int64_t>(data.size())}, intType);
    memref::GlobalOp::create(b, name, b.getStringAttr("private"), memrefType,
                             denseAttr, /*constant=*/true,
                             /*alignment=*/IntegerAttr{});
  }
  auto memrefType =
      MemRefType::get({static_cast<int64_t>(data.size())}, intType);
  return memref::GetGlobalOp::create(b, memrefType, name);
}

Value getOrCreatePProdGlobal(ImplicitLocOpBuilder& b, ModuleOp module,
                             const cheddar_ref::Params64& params, int level,
                             Type intType) {
  std::string name = "__cheddar_pseudo_mod_up_p_prod_l" + std::to_string(level);
  auto tbl = cheddar_ref::ModSwitchTables64::GeneratePseudoModUp(params, level);
  return getOrCreate1DGlobal<uint64_t>(b, module, name, tbl.p_prod, intType);
}

struct BootConstsEntry {
  cheddar_ref::Params<64> params;
  cheddar_ref::Tables<64> tables;
  std::unique_ptr<cheddar_ref::Encoder<64>> encoder;
  std::unique_ptr<cheddar_ref::BootConsts<64>> bootConsts;
};

struct BootConstsCache {
  DenseMap<ParamsAttr, std::unique_ptr<BootConstsEntry>> entries;
  DenseMap<std::pair<ParamsAttr, int64_t>, cheddar_ref::LinearTransformPlan>
      ltPlanMap;
  DenseMap<std::pair<ParamsAttr, int64_t>, cheddar_ref::BootEvkPlan> evkPlanMap;

  cheddar_ref::BootConsts64* getBootConsts(
      ParamsAttr paramsAttr, const cheddar_ref::Params<64>& params) {
    auto it = entries.find(paramsAttr);
    if (it != entries.end()) return it->second->bootConsts.get();
    auto entry = std::make_unique<BootConstsEntry>();
    entry->params = params;
    entry->tables =
        cheddar_ref::Tables64::Generate(entry->params, /*max_limbs=*/0);
    entry->encoder =
        std::make_unique<cheddar_ref::Encoder64>(entry->params, entry->tables);
    entry->bootConsts = std::make_unique<cheddar_ref::BootConsts64>(
        entry->params, entry->tables, *entry->encoder);
    auto* ptr = entry->bootConsts.get();
    entries[paramsAttr] = std::move(entry);
    return ptr;
  }

  const cheddar_ref::LinearTransformPlan& getLtPlan(
      ParamsAttr paramsAttr, const cheddar_ref::Params<64>& params,
      int64_t numSlots) {
    auto key = std::make_pair(paramsAttr, numSlots);
    auto it = ltPlanMap.find(key);
    if (it != ltPlanMap.end()) return it->second;
    auto* bc = getBootConsts(paramsAttr, params);
    ltPlanMap[key] = bc->GetLinearTransformPlan(numSlots);
    return ltPlanMap[key];
  }

  const cheddar_ref::BootEvkPlan& getEvkPlan(
      ParamsAttr paramsAttr, const cheddar_ref::Params<64>& params,
      int64_t numSlots) {
    auto key = std::make_pair(paramsAttr, numSlots);
    auto it = evkPlanMap.find(key);
    if (it != evkPlanMap.end()) return it->second;
    auto* bc = getBootConsts(paramsAttr, params);
    evkPlanMap[key] = bc->GetBootEvkPlan(numSlots);
    return evkPlanMap[key];
  }
};

struct ExpandHoistPhasePattern : public OpRewritePattern<HoistPhaseOp> {
  ExpandHoistPhasePattern(MLIRContext* context, BootConstsCache& cache)
      : OpRewritePattern<HoistPhaseOp>(context), cache_(cache) {}

  LogicalResult matchAndRewrite(HoistPhaseOp op,
                                PatternRewriter& rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module) return op.emitOpError("missing parent module");

    ParamsAttr paramsAttr;
    if (op.getParams()) {
      paramsAttr = *op.getParams();
    } else {
      paramsAttr = module->getAttrOfType<ParamsAttr>("cheddar_gpu.params");
    }
    if (!paramsAttr) {
      return op.emitOpError("missing cheddar_gpu.params attribute");
    }

    uint32_t word = paramsAttr.getWord();
    if (word != 64) {
      return op.emitOpError(
          "only 64-bit word size is supported for hoist expansion");
    }

    int64_t numSlots = op.getNumSlots();
    StringRef kind = op.getKind();
    int64_t phase = op.getPhase();

    auto params = paramsFromAttr<64>(paramsAttr);
    const auto& ltPlan = cache_.getLtPlan(paramsAttr, params, numSlots);
    const auto& evkPlan = cache_.getEvkPlan(paramsAttr, params, numSlots);

    const cheddar_ref::LinearTransformPhaseSpec* phaseSpec = nullptr;
    if (kind == "cts") {
      if (static_cast<size_t>(phase) >= ltPlan.cts.size()) {
        return op.emitOpError("phase out of range for cts");
      }
      phaseSpec = &ltPlan.cts[phase];
    } else {
      if (static_cast<size_t>(phase) >= ltPlan.stc.size()) {
        return op.emitOpError("phase out of range for stc");
      }
      phaseSpec = &ltPlan.stc[phase];
    }

    std::string fnName =
        llvm::formatv("__cheddar_hoist_{0}{1}_s{2}", kind, phase, numSlots)
            .str();
    auto fnOp = module.lookupSymbol<func::FuncOp>(fnName);
    if (!fnOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());

      auto funcType = FunctionType::get(
          op.getContext(),
          {op.getOutBx().getType(), op.getOutAx().getType(),
           op.getInBx().getType(), op.getInAx().getType(),
           op.getConstsSlab().getType(), op.getEvkSlab().getType()},
          {});
      fnOp = func::FuncOp::create(rewriter, op.getLoc(), fnName, funcType);
      fnOp.setPrivate();

      Block* entryBlock = fnOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);
      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      Value func_out_bx = entryBlock->getArgument(0);
      Value func_out_ax = entryBlock->getArgument(1);
      Value func_in_bx = entryBlock->getArgument(2);
      Value func_in_ax = entryBlock->getArgument(3);
      Value func_consts_slab = entryBlock->getArgument(4);
      Value func_evk_slab = entryBlock->getArgument(5);

      int pt_level = phaseSpec->pt_level;
      int64_t num_q = pt_level + 1;
      int64_t num_aux = paramsAttr.getAlpha();
      int64_t num_qp = num_q + num_aux;
      int64_t degree = params.degree;
      Type intType = b.getI64Type();
      auto qType = MemRefType::get({num_q, degree}, intType);
      auto qpType = MemRefType::get({num_qp, degree}, intType);

      // (1) bs_slab[0] = pseudo-mod-up of in (mul_const by p_prod[pt_level] on
      // Q limbs, zero_pad on P limbs)
      Value pProd =
          getOrCreatePProdGlobal(b, module, params, pt_level, intType);

      Value bs0_bx_q = memref::AllocOp::create(b, qType);
      Value bs0_ax_q = memref::AllocOp::create(b, qType);
      MulConstOp::create(b, bs0_bx_q, func_in_bx, pProd, Form::Eval,
                         Form::Eval);
      MulConstOp::create(b, bs0_ax_q, func_in_ax, pProd, Form::Eval,
                         Form::Eval);

      Value bs0_bx = memref::AllocOp::create(b, qpType);
      Value bs0_ax = memref::AllocOp::create(b, qpType);
      CopyOp::create(b, bs0_bx, bs0_bx_q,
                     /*in_limb_range=*/b.getDenseI64ArrayAttr({0, num_q}),
                     /*out_limb_range=*/b.getDenseI64ArrayAttr({0, num_q}));
      CopyOp::create(b, bs0_ax, bs0_ax_q,
                     /*in_limb_range=*/b.getDenseI64ArrayAttr({0, num_q}),
                     /*out_limb_range=*/b.getDenseI64ArrayAttr({0, num_q}));
      ZeroPadOp::create(b, bs0_bx, b.getDenseI64ArrayAttr({num_q, num_qp}));
      ZeroPadOp::create(b, bs0_ax, b.getDenseI64ArrayAttr({num_q, num_qp}));

      // (2) mod_up of in.ax -> beta digits
      int64_t prime_offset = 0;
      int64_t beta = (num_q + prime_offset + num_aux - 1) / num_aux;
      SmallVector<Value> mod_up_digits;
      mod_up_digits.reserve(beta);
      for (int64_t d = 0; d < beta; ++d) {
        mod_up_digits.push_back(memref::AllocOp::create(b, qpType));
      }
      ModUpOp::create(b, mod_up_digits, func_in_ax, pt_level, Form::Eval,
                      Form::Eval);

      // (3) ONE cheddar_gpu.hoist_bs_fused producing all R = num_bs - 1 rotated
      // baby-step ciphertexts
      size_t num_bs = phaseSpec->bs_indices.size();
      SmallVector<Value> bs_out_bx, bs_out_ax;
      if (num_bs > 1) {
        size_t R = num_bs - 1;
        SmallVector<int64_t> galois_factors;
        SmallVector<int64_t> evk_byte_offsets;
        SmallVector<int64_t> evk_num_q;
        galois_factors.reserve(R);
        evk_byte_offsets.reserve(R);
        evk_num_q.reserve(R);

        for (size_t r = 0; r < R; ++r) {
          bs_out_bx.push_back(memref::AllocOp::create(b, qpType));
          bs_out_ax.push_back(memref::AllocOp::create(b, qpType));

          int rot = phaseSpec->bs_indices[r + 1];
          galois_factors.push_back(
              computeGaloisFactor(degree / 2 - rot, degree));

          const cheddar_ref::BootEvkSpec* spec = nullptr;
          for (const auto& evk : evkPlan.evks) {
            if (evk.kind == cheddar_ref::BootEvkKind::kRot &&
                evk.rot_index == rot) {
              spec = &evk;
              break;
            }
          }
          assert(spec != nullptr && "Missing EVK for rotation step");
          evk_byte_offsets.push_back(spec->slab_byte_offset);
          evk_num_q.push_back(spec->num_q);
        }

        HoistBSFusedOp::create(
            b, bs_out_bx, bs_out_ax, mod_up_digits, func_evk_slab, bs0_bx,
            b.getDenseI64ArrayAttr(galois_factors),
            b.getDenseI64ArrayAttr(evk_byte_offsets),
            b.getDenseI64ArrayAttr(evk_num_q), b.getI64IntegerAttr(num_q),
            b.getI64IntegerAttr(num_aux), b.getI64IntegerAttr(prime_offset));
      }

      // (4) ONE cheddar_gpu.hoist_gs_fused producing all G giant-step
      // accumulators from the B baby steps
      size_t G = phaseSpec->gs_indices.size();
      SmallVector<Value> all_bs_bx, all_bs_ax;
      all_bs_bx.push_back(bs0_bx);
      all_bs_ax.push_back(bs0_ax);
      for (Value v : bs_out_bx) all_bs_bx.push_back(v);
      for (Value v : bs_out_ax) all_bs_ax.push_back(v);

      SmallVector<Value> gs_out_bx, gs_out_ax;
      for (size_t g = 0; g < G; ++g) {
        gs_out_bx.push_back(memref::AllocOp::create(b, qpType));
        gs_out_ax.push_back(memref::AllocOp::create(b, qpType));
      }

      SmallVector<int64_t> pt_word_offsets;
      pt_word_offsets.reserve(G * num_bs);
      for (size_t g = 0; g < G; ++g) {
        for (size_t b_idx = 0; b_idx < num_bs; ++b_idx) {
          pt_word_offsets.push_back(phaseSpec->pt_slab_word_offset[g][b_idx]);
        }
      }

      HoistGSFusedOp::create(
          b, gs_out_bx, gs_out_ax, all_bs_bx, all_bs_ax, func_consts_slab,
          b.getDenseI64ArrayAttr(pt_word_offsets), b.getI64IntegerAttr(num_q),
          b.getI64IntegerAttr(num_aux), b.getI64IntegerAttr(prime_offset),
          b.getI64IntegerAttr(num_qp));

      // (5) giant-step loop: final = gs[0] (or zero if gs index 0 absent)
      int64_t gs0_idx = -1;
      for (size_t g = 0; g < G; ++g) {
        if (phaseSpec->gs_indices[g] == 0) {
          gs0_idx = static_cast<int64_t>(g);
          break;
        }
      }

      Value final_bx, final_ax;
      if (gs0_idx >= 0) {
        final_bx = gs_out_bx[gs0_idx];
        final_ax = gs_out_ax[gs0_idx];
      } else {
        final_bx = memref::AllocOp::create(b, qpType);
        final_ax = memref::AllocOp::create(b, qpType);
        ZeroPadOp::create(b, final_bx, b.getDenseI64ArrayAttr({0, num_qp}));
        ZeroPadOp::create(b, final_ax, b.getDenseI64ArrayAttr({0, num_qp}));
      }

      bool first_no_gs0 = (gs0_idx < 0);
      for (size_t g = 0; g < G; ++g) {
        int gs_idx = phaseSpec->gs_indices[g];
        if (gs_idx == 0) continue;

        Value tmp_moddown = memref::AllocOp::create(b, qType);
        ModDownOp::create(b, tmp_moddown, gs_out_ax[g], pt_level, Form::Eval,
                          Form::Eval);

        const cheddar_ref::BootEvkSpec* spec = nullptr;
        for (const auto& evk : evkPlan.evks) {
          if (evk.kind == cheddar_ref::BootEvkKind::kRot &&
              evk.rot_index == gs_idx) {
            spec = &evk;
            break;
          }
        }
        assert(spec != nullptr && "Missing EVK for giant-step rotation");

        int64_t evk_poly_bytes =
            (spec->num_q + spec->num_aux) * degree * sizeof(uint64_t);
        auto evkDigitType =
            MemRefType::get({spec->num_q + spec->num_aux, degree}, intType);
        SmallVector<Value> evkBx, evkAx;
        evkBx.reserve(spec->beta);
        evkAx.reserve(spec->beta);
        for (int64_t d = 0; d < spec->beta; ++d) {
          int64_t bxOffset = spec->slab_byte_offset + d * 2 * evk_poly_bytes;
          int64_t axOffset = bxOffset + evk_poly_bytes;
          Value bxOffVal = arith::ConstantIndexOp::create(b, bxOffset);
          Value axOffVal = arith::ConstantIndexOp::create(b, axOffset);
          evkBx.push_back(memref::ViewOp::create(b, evkDigitType, func_evk_slab,
                                                 bxOffVal, ValueRange{}));
          evkAx.push_back(memref::ViewOp::create(b, evkDigitType, func_evk_slab,
                                                 axOffVal, ValueRange{}));
        }

        Value ks_bx = memref::AllocOp::create(b, qpType);
        Value ks_ax = memref::AllocOp::create(b, qpType);
        KeySwitchNoModDownOp::create(
            b, ks_bx, ks_ax, tmp_moddown, /*in_bx=*/Value(), /*in_ax=*/Value(),
            evkBx, evkAx, pt_level, Form::Eval, Form::Eval);

        int64_t galois = computeGaloisFactor(gs_idx, degree);
        Value perm_bx = memref::AllocOp::create(b, qpType);
        Value perm_ax = memref::AllocOp::create(b, qpType);
        Permute2Op::create(b, perm_bx, perm_ax, ks_bx, ks_ax, Form::Eval,
                           Form::Eval, b.getI64IntegerAttr(galois),
                           /*conj=*/false, DenseI64ArrayAttr{},
                           DenseI64ArrayAttr{}, b.getI64IntegerAttr(num_aux));

        if (first_no_gs0) {
          final_bx = perm_bx;
          final_ax = perm_ax;
          first_no_gs0 = false;
        } else {
          Value next_final_bx = memref::AllocOp::create(b, qpType);
          Value next_final_ax = memref::AllocOp::create(b, qpType);
          Add2Op::create(b, next_final_bx, next_final_ax, final_bx, final_ax,
                         perm_bx, perm_ax, Form::Eval, Form::Eval, Form::Eval,
                         DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
                         DenseI64ArrayAttr{}, b.getI64IntegerAttr(num_aux));
          final_bx = next_final_bx;
          final_ax = next_final_ax;
        }
      }

      // PermuteAccum of all gs[k].bx into final.bx
      SmallVector<Value> rot_bx_srcs;
      SmallVector<int64_t> rot_galois;
      for (size_t g = 0; g < G; ++g) {
        int gs_idx = phaseSpec->gs_indices[g];
        if (gs_idx == 0) continue;
        rot_bx_srcs.push_back(gs_out_bx[g]);
        rot_galois.push_back(computeGaloisFactor(gs_idx, degree));
      }

      if (!rot_bx_srcs.empty()) {
        if (rot_bx_srcs.size() <= 8) {
          Value new_final_bx = memref::AllocOp::create(b, qpType);
          PermuteAccumOp::create(
              b, new_final_bx, final_bx, rot_bx_srcs, Form::Eval, ArrayAttr{},
              b.getDenseI64ArrayAttr(rot_galois), DenseI64ArrayAttr{},
              IntegerAttr{}, b.getI64IntegerAttr(num_aux));
          final_bx = new_final_bx;
        } else {
          size_t processed = 0;
          while (processed < rot_bx_srcs.size()) {
            size_t chunk_len =
                std::min<size_t>(8, rot_bx_srcs.size() - processed);
            SmallVector<Value> chunk_srcs(
                rot_bx_srcs.begin() + processed,
                rot_bx_srcs.begin() + processed + chunk_len);
            SmallVector<int64_t> chunk_galois(
                rot_galois.begin() + processed,
                rot_galois.begin() + processed + chunk_len);
            Value new_final_bx = memref::AllocOp::create(b, qpType);
            PermuteAccumOp::create(
                b, new_final_bx, final_bx, chunk_srcs, Form::Eval, ArrayAttr{},
                b.getDenseI64ArrayAttr(chunk_galois), DenseI64ArrayAttr{},
                IntegerAttr{}, b.getI64IntegerAttr(num_aux));
            final_bx = new_final_bx;
            processed += chunk_len;
          }
        }
      }

      // (6) final mod_down with rescale -> out at pt_level - 1
      ModDownRescaleOp::create(b, func_out_bx, final_bx, pt_level, Form::Eval,
                               Form::Eval);
      ModDownRescaleOp::create(b, func_out_ax, final_ax, pt_level, Form::Eval,
                               Form::Eval);
      func::ReturnOp::create(b);
    }

    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fnOp,
        ValueRange{op.getOutBx(), op.getOutAx(), op.getInBx(), op.getInAx(),
                   op.getConstsSlab(), op.getEvkSlab()});
    return success();
  }

 private:
  BootConstsCache& cache_;
};

struct ExpandHoist : public impl::ExpandHoistBase<ExpandHoist> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* context = &getContext();
    BootConstsCache cache;
    RewritePatternSet patterns(context);
    patterns.add<ExpandHoistPhasePattern>(context, cache);
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createExpandHoistPass() {
  return std::make_unique<ExpandHoist>();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
