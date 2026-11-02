#ifndef LIB_UTILS_CHEDDARREF_BOOTCONSTS_H_
#define LIB_UTILS_CHEDDARREF_BOOTCONSTS_H_

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "lib/Utils/CheddarRef/Encoder.h"
#include "lib/Utils/CheddarRef/Params.h"
#include "lib/Utils/CheddarRef/Tables.h"

namespace mlir::heir::cheddar_ref {

struct BootScalars {
  int32_t log_scaleup = 0;
  int32_t initial_K = 2;
  int32_t num_double_angle = 3;
  int32_t max_level = 0;
  int32_t cts_start_level = 0;
  int32_t eval_mod_start_level = 0;
  int32_t stc_start_level = 0;
  int32_t end_level = 0;
  double cts_const = 0.0;
  double stc_const = 0.0;
};

/// Represents a striped matrix indexed by diagonal offset -> vector of
/// elements.
using StripedMat = std::map<int, std::vector<std::complex<double>>>;

/// Constants for AXYPBZ evaluation: a * x * y + b * z (or a * x * y + b).
template <unsigned W>
struct AXYPBZConsts {
  using Word = typename Tables<W>::Word;
  int x_level = 0;
  double x_scale = 0.0;
  int y_level = 0;
  double y_scale = 0.0;
  int z_level = 0;
  double z_scale = 0.0;
  int final_level = 0;
  double final_scale = 0.0;
  bool has_a = false;
  bool has_b = false;
  bool has_z = false;
  std::vector<Word> a_cx;
  std::vector<Word> b_cx;
};

enum class EvalPolyType { kNormal, kOdd, kEven };

template <unsigned W>
class RefBasisMap {
 public:
  RefBasisMap(const Params<W>& params, const Encoder<W>& encoder,
              int input_level, double input_scale,
              EvalPolyType type = EvalPolyType::kNormal);

  int SplitBaseDegree(int base_degree) const;
  bool Exists(int base_degree) const;
  std::pair<int, double> GetBaseLevelAndScale(int base_degree) const;
  bool IsMultUnsafeCompatible(int level1, int level2) const;
  void AddBase(int base_degree);

  const std::map<int, AXYPBZConsts<W>>& GetBasisEval() const {
    return basis_eval_;
  }
  int GetInputLevel() const { return input_level_; }
  double GetInputScale() const { return input_scale_; }
  EvalPolyType GetType() const { return type_; }

 private:
  const Params<W>& params_;
  const Encoder<W>& encoder_;
  int input_level_;
  double input_scale_;
  EvalPolyType type_;
  std::map<int, AXYPBZConsts<W>> basis_eval_;
};

template <unsigned W>
class RefEvalPolyNode {
 public:
  using Word = typename Tables<W>::Word;

  RefEvalPolyNode(const std::vector<double>& coefficients, int level_margin,
                  int baby_threshold);

  void CheckRequiredBasis(std::set<int>& required_bases) const;
  void Compile(const Encoder<W>& encoder, const RefBasisMap<W>& basis_map,
               int target_level, double target_scale, bool do_rescale = true);
  void CollectLimbs(std::vector<Word>& limbs);

  bool IsLeaf() const {
    return low_ == nullptr && high_ == nullptr && !is_high_constant_ &&
           !is_low_constant_;
  }
  bool IsInterior() const { return !IsLeaf(); }

  int GetTargetLevel() const { return target_level_; }
  double GetTargetScale() const { return target_scale_; }
  bool GetDoRescale() const { return do_rescale_; }
  int GetSplitDegree() const { return split_degree_; }
  bool IsLowZero() const { return is_low_zero_; }
  bool IsLowConstant() const { return is_low_constant_; }
  bool IsHighConstant() const { return is_high_constant_; }

  std::shared_ptr<RefEvalPolyNode<W>> GetLow() const { return low_; }
  std::shared_ptr<RefEvalPolyNode<W>> GetHigh() const { return high_; }

  int64_t GetLowConstOffset() const { return low_const_offset_; }
  int64_t GetHighConstOffset() const { return high_const_offset_; }
  const std::map<int, int64_t>& GetLeafConstOffsets() const {
    return leaf_const_offsets_;
  }

  const std::vector<Word>& GetLowConstant() const { return low_constant_; }
  const std::vector<Word>& GetHighConstant() const { return high_constant_; }
  const std::map<int, std::vector<Word>>& GetLeafConstants() const {
    return leaf_constants_;
  }
  const std::vector<double>& GetCoefficients() const { return coefficients_; }

 private:
  std::vector<double> coefficients_;
  int split_degree_ = 0;
  bool is_low_zero_ = false;
  bool is_low_constant_ = false;
  bool is_high_constant_ = false;

  int target_level_ = 0;
  double target_scale_ = 0.0;
  bool do_rescale_ = true;

  int64_t low_const_offset_ = -1;
  int64_t high_const_offset_ = -1;
  std::map<int, int64_t> leaf_const_offsets_;

  std::vector<Word> low_constant_;
  std::vector<Word> high_constant_;
  std::map<int, std::vector<Word>> leaf_constants_;

  std::shared_ptr<RefEvalPolyNode<W>> low_;
  std::shared_ptr<RefEvalPolyNode<W>> high_;
};

/// Linear transform phase specification for compile-time query (zero NTT).
struct LinearTransformPhaseSpec {
  int pt_level = 0;
  double pt_scale = 0.0;
  int bs = 0;
  int gs = 0;
  std::vector<int> bs_indices;
  std::vector<int> gs_indices;
  bool bs_gs_swapped = false;
  int pre_rotation = 0;
  int additional_pt_rot = 0;
  std::vector<std::vector<int64_t>>
      pt_slab_word_offset; /*[gs][bs], -1 if absent*/
  int num_limbs = 0;
};

/// Linear transform plan covering CtS and StC phases.
struct LinearTransformPlan {
  std::vector<LinearTransformPhaseSpec> cts;
  std::vector<LinearTransformPhaseSpec> stc;
  int64_t total_slab_words = 0;
};

/// EVK kind indicator.
enum class BootEvkKind {
  kDts,
  kStd,
  kConj,
  kRelin,
  kRot,
};

/// Specification of a single evaluation key in the boot EVK slab.
struct BootEvkSpec {
  BootEvkKind kind = BootEvkKind::kRot;
  int rot_index = 0;
  int max_level = 0;
  int num_q = 0;
  int num_aux = 0;
  int beta = 0;
  int64_t slab_byte_offset = 0;
  int64_t total_bytes = 0;
};

/// Full EVK plan for bootstrapping.
struct BootEvkPlan {
  std::vector<BootEvkSpec> evks;
  int64_t total_slab_bytes = 0;
};

template <unsigned W>
class BootConsts {
 public:
  using Word = typename Tables<W>::Word;
  using Complex = std::complex<double>;

  BootConsts(const Params<W>& params, const Tables<W>& tables,
             const Encoder<W>& encoder, int num_cts_levels = 4,
             int num_stc_levels = 3, int log_message_ratio = 5,
             int num_double_angle = 3, int initial_K = 2);

  const BootScalars& GetScalars() const { return scalars_; }

  /// Constant for scale adjustment at level -1.
  const std::vector<Word>& GetScaleupConst() const { return scaleup_const_; }

  /// ModMax INTT constants for q0.
  const std::vector<Word>& GetModMaxInttConst() const {
    return mod_max_intt_const_;
  }

  /// EvalMod initial constant: -0.25 / actual_K at start_level, start_scale.
  const std::vector<Word>& GetEvalModInitialConst() const {
    return evalmod_initial_const_;
  }

  /// Double angle constants (for each step).
  const std::vector<AXYPBZConsts<W>>& GetDoubleAngleConsts() const {
    return double_angle_;
  }

  /// EvalPoly collected constants (limbs matching PolyConstCollector order).
  const std::vector<Word>& GetEvalPolyConsts() const {
    return evalpoly_consts_;
  }

  /// Basis map computed during InitEvalMod.
  const RefBasisMap<W>& GetBasisMap() const { return *basis_map_; }

  /// EvalPoly root node computed during InitEvalMod.
  std::shared_ptr<RefEvalPolyNode<W>> GetEvalPolyRoot() const {
    return evalpoly_root_;
  }

  /// Context::LevelDown constants (levels 1 .. default_encryption_level).
  const std::vector<std::vector<Word>>& GetLevelDownConsts() const {
    return level_down_consts_;
  }

  /// Imaginary unit constants: psi_rev_mont[l*N+1] for each prime.
  const std::vector<Word>& GetImaginaryUnitConsts() const {
    return imaginary_unit_consts_;
  }

  /// StripedMatrix operations.
  static StripedMat MultStriped(const StripedMat& a, const StripedMat& b,
                                int width);
  static StripedMat ScaleStriped(const StripedMat& a, Complex scalar);

  /// Generate plain special FFT/IFFT stages (StripedMatrix per stage).
  std::vector<StripedMat> GenerateIFFTStages(int num_slots) const;
  std::vector<StripedMat> GenerateFFTStages(int num_slots) const;

  /// Phase matrices for CtS and StC.
  std::vector<StripedMat> GenerateCtSPhaseMatrices(int num_slots) const;
  std::vector<StripedMat> GenerateStCPhaseMatrices(int num_slots) const;

  /// Hoist plain map: [gs_idx][bs_idx] -> rotated message
  using PlainHoistMap = std::map<int, std::map<int, std::vector<Complex>>>;

  /// BSGS split matching Cheddar's EvalSpecialFFT::BSGSSplit
  static std::pair<int, int> BSGSSplit(int num_diag);

  /// Constructs plain hoist map matching Cheddar's
  /// LinearTransform::ConstructPlainHoistMap
  static PlainHoistMap ConstructPlainHoistMap(const StripedMat& matrix, int bs,
                                              int gs, int pre_rotation,
                                              int additional_pt_rot);

  /// Encode a complex diagonal into a Plaintext polynomial in NTT domain.
  std::vector<Word> EncodeDiagonal(const std::vector<Complex>& diag,
                                   int pt_level, double pt_scale,
                                   int num_aux) const;

  /// Compile-time linear transform plan (zero NTT operations).
  LinearTransformPlan GetLinearTransformPlan(int num_slots) const;

  /// Bootstrapping EVK plan (required rotation and special EVK slab layout).
  BootEvkPlan GetBootEvkPlan(int num_slots) const;

  /// Packs all CtS and StC encoded plaintexts into a single device slab.
  std::vector<Word> PackBootConstsSlab(int num_slots) const;

 private:
  const Params<W>& params_;
  const Tables<W>& tables_;
  const Encoder<W>& encoder_;

  int num_cts_levels_;
  int num_stc_levels_;
  int log_message_ratio_;
  int num_double_angle_;
  int initial_K_;

  BootScalars scalars_;
  std::vector<Word> scaleup_const_;
  std::vector<Word> mod_max_intt_const_;
  std::vector<Word> evalmod_initial_const_;
  std::vector<AXYPBZConsts<W>> double_angle_;
  std::vector<Word> evalpoly_consts_;

  std::shared_ptr<RefBasisMap<W>> basis_map_;
  std::shared_ptr<RefEvalPolyNode<W>> evalpoly_root_;
  std::vector<std::vector<Word>> level_down_consts_;
  std::vector<Word> imaginary_unit_consts_;

  void InitScalarsAndContextConsts();
  void InitEvalMod();
};

using BootConsts32 = BootConsts<32>;
using BootConsts64 = BootConsts<64>;

}  // namespace mlir::heir::cheddar_ref

#endif  // LIB_UTILS_CHEDDARREF_BOOTCONSTS_H_
