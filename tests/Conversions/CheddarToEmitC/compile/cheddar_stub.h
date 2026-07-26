// Header-only stub of the CHEDDAR C++ API, used to *compile* (not run) the
// C++ that `cheddar-to-emitc` + `heir-translate --mlir-to-cpp` produce, with
// no GPU/CUDA toolchain. This is a CI-runnable guard that the emitted code
// honours CHEDDAR's move/const contract -- the kind of bug that FileCheck
// (which only inspects emitted text) cannot catch.
//
// The move/const semantics below mirror the real library (verified against
// CHEDDAR's include/core headers); only these properties matter here, so the
// method bodies are empty and the data layout is omitted:
//
//   * Ciphertext/Plaintext/Constant/EvaluationKey -- move-only (copy deleted)
//     *with* move-assignment, default-constructible.   (core/Container.h)
//   * EvkMap -- move-only, copy deleted, *no* move-assignment. (core/EvkMap.h)
//   * Context::MadUnsafe(Ct& res, ...) mutates `res` in place, so `res` is a
//     non-const reference.                              (core/Context.h:377)
//   * UserInterface::Get*Key() / GetEvkMap() return `const&`. (UserInterface.h)
//
// Kept deliberately narrow: the "setup/getter" surface (create_context,
// create_user_interface, get_encoder, encode/decode) is *not* modelled here
// because that part of the emitter has independent, pre-existing mismatches
// against the real API and needs its own design pass.

#ifndef TESTS_CONVERSIONS_CHEDDARTOEMITC_COMPILE_CHEDDAR_STUB_H_
#define TESTS_CONVERSIONS_CHEDDARTOEMITC_COMPILE_CHEDDAR_STUB_H_

#include <complex>
#include <initializer_list>
#include <map>
#include <memory>
#include <vector>

namespace cheddar {

using Complex = std::complex<double>;

// Minimal parameter stub: the LinearTransform emitter reads the per-level
// canonical scale via `ctx->param_.GetScale(level)`.
template <typename word>
struct Parameter {
  double GetScale(int level) const;
};

// Move-only payload types with full move support (default + move-ctor +
// move-assign; copy deleted).
template <typename word>
struct Ciphertext {
  Ciphertext() = default;
  Ciphertext(Ciphertext&&) = default;
  Ciphertext& operator=(Ciphertext&&) = default;
  Ciphertext(const Ciphertext&) = delete;
  Ciphertext& operator=(const Ciphertext&) = delete;
};
template <typename word>
struct Plaintext {
  Plaintext() = default;
  Plaintext(Plaintext&&) = default;
  Plaintext& operator=(Plaintext&&) = default;
  Plaintext(const Plaintext&) = delete;
  Plaintext& operator=(const Plaintext&) = delete;
};
template <typename word>
struct Constant {
  Constant() = default;
  Constant(Constant&&) = default;
  Constant& operator=(Constant&&) = default;
  Constant(const Constant&) = delete;
  Constant& operator=(const Constant&) = delete;
};
template <typename word>
struct EvaluationKey {
  EvaluationKey() = default;
  EvaluationKey(EvaluationKey&&) = default;
  EvaluationKey& operator=(EvaluationKey&&) = default;
  EvaluationKey(const EvaluationKey&) = delete;
  EvaluationKey& operator=(const EvaluationKey&) = delete;
};

// Move-only, and -- unlike the payload types -- has *no* move-assignment and
// is not default-constructible (the real EvkMap inherits std::unordered_map
// and declares only a move ctor). This is what makes the value+assign getter
// shape uncompilable, so the stub preserves it.
template <typename word>
struct EvkMap {
  EvkMap(EvkMap&&) = default;
  EvkMap(const EvkMap&) = delete;
  EvkMap& operator=(const EvkMap&) = delete;

  const EvaluationKey<word>& GetRotationKey(int) const;
  const EvaluationKey<word>& GetConjugationKey() const;
  const EvaluationKey<word>& GetMultiplicationKey() const;
};

template <typename word>
class UserInterface {
 public:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  void Encrypt(Ct& res, const Pt& a) const;
  void Decrypt(Pt& res, const Ct& a) const;

  const Evk& GetRotationKey(int rot_idx) const;
  const Evk& GetConjugationKey() const;
  const Evk& GetMultiplicationKey() const;
  const EvkMap<word>& GetEvkMap() const;
};

template <typename word>
class Context {
 public:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Const = Constant<word>;
  using Evk = EvaluationKey<word>;

  // Overloaded ct/pt/const arithmetic -- the emitter dispatches the dialect's
  // *_plain / *_const ops to the base name and relies on C++ overloading.
  void Add(Ct& res, const Ct& a, const Ct& b) const;
  void Add(Ct& res, const Ct& a, const Pt& b) const;
  void Add(Ct& res, const Ct& a, const Const& b) const;
  void Sub(Ct& res, const Ct& a, const Ct& b) const;
  void Sub(Ct& res, const Ct& a, const Pt& b) const;
  void Mult(Ct& res, const Ct& a, const Ct& b) const;
  void Mult(Ct& res, const Ct& a, const Pt& b) const;
  void Mult(Ct& res, const Ct& a, const Const& b) const;

  void Neg(Ct& res, const Ct& a) const;
  void Rescale(Ct& res, const Ct& a) const;
  void Relinearize(Ct& res, const Ct& a, const Evk& key) const;
  void RelinearizeRescale(Ct& res, const Ct& a, const Evk& key) const;
  void LevelDown(Ct& res, const Ct& a, int target_level) const;

  void HMult(Ct& res, const Ct& a, const Ct& b, const Evk& mult_key,
             bool rescale) const;
  void HRot(Ct& res, const Ct& a, const Evk& rot_key, int rot_dist) const;
  void HRotAdd(Ct& res, const Ct& a, const Ct& b, const Evk& rot_key,
               int rot_dist) const;
  void HConj(Ct& res, const Ct& a, const Evk& conj_key) const;
  void HConjAdd(Ct& res, const Ct& a, const Ct& b, const Evk& conj_key) const;

  // In-place multiply-accumulate: `res` is mutated, so it is a *non-const*
  // reference. This is the crux of the mad_unsafe finding.
  void MadUnsafe(Ct& res, const Ct& a, const Const& b) const;

  // The LinearTransform / EvalPoly emitters read the canonical scale from here.
  Parameter<word> param_;
};

// CHEDDAR's LinearTransform extension is a class constructed from a
// StripedMatrix (diagonal index -> diagonal) and Evaluate()'d; the emitter
// builds one inline. ConstContextPtr is a non-owning shared_ptr aliased onto
// the raw Context*.
template <typename word>
using ConstContextPtr = std::shared_ptr<const Context<word>>;

class StripedMatrix : public std::map<int, std::vector<Complex>> {
 public:
  StripedMatrix(int height = 0, int width = 0) {}
};

template <typename word>
class LinearTransform {
 public:
  LinearTransform(ConstContextPtr<word> context, const StripedMatrix& matrix,
                  int pt_level, double pt_scale, int bs, int gs = 1,
                  int pre_rotation = 0, int additional_pt_rot = 0);
  void Evaluate(ConstContextPtr<word> context, Ciphertext<word>& res,
                const Ciphertext<word>& input, const EvkMap<word>& evk_map,
                bool min_ks = false) const;
};

// CHEDDAR's EvalPoly extension: also a class -- construct from coefficients,
// Compile(), then Evaluate() with the multiplication key.
template <typename word>
class EvalPoly {
 public:
  EvalPoly(const std::vector<double>& coefficients, int input_level,
           double input_scale, double target_scale, bool chebyshev = false);
  void Compile(ConstContextPtr<word> context);
  void Evaluate(ConstContextPtr<word> context, Ciphertext<word>& res,
                const Ciphertext<word>& input,
                const EvaluationKey<word>& mult_key) const;
};

// Boot lives on the derived BootContext (extension/BootContext.h), not Context.
// cheddar.boot takes a !cheddar.boot_context, lowered to BootContext<word>*, so
// the emitter calls `ctx->Boot(...)` directly; the stub mirrors that hierarchy.
template <typename word>
class BootContext : public Context<word> {
 public:
  using Ct = Ciphertext<word>;
  void Boot(Ct& res, const Ct& a, const EvkMap<word>& evk_map) const;
};

}  // namespace cheddar

#endif  // TESTS_CONVERSIONS_CHEDDARTOEMITC_COMPILE_CHEDDAR_STUB_H_
