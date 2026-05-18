#ifndef LIB_TARGET_JAXITEWORD_JAXITEWORDTEMPLATES_H_
#define LIB_TARGET_JAXITEWORD_JAXITEWORDTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace jaxiteword {

constexpr std::string_view kModulePrelude = R"python(
import jax
import jax.numpy as jnp
import numpy as np
from ciphertext import Ciphertext
from polynomial import Polynomial
import ckks_ctx as ckks

)python";

// Template for GenMulKeyOp
// This template initializes HEMul for homomorphic multiplication and sets up
// relinearization. It computes r and c from the degree if they are not provided
// in the parameters.
constexpr std::string_view kGenMulKeyTemplate = R"python(
_degree = {0}.parameters.get('degree')
if _degree is not None:
  _log_degree = int(math.log2(_degree))
  _half_k = _log_degree // 2
  _default_r = 2 ** _half_k
  _default_c = _degree // _default_r
else:
  _default_r = 4
  _default_c = 4
{1} = HEMul(
    batch={0}.parameters.get('batch', 1),
    r={0}.parameters.get('r', _default_r),
    c={0}.parameters.get('c', _default_c),
    dnum={0}.parameters.get('dnum', 3),
    num_eval_mult={0}.parameters.get('numEvalMult', 1),
    original_moduli={0}.q_towers,
    extend_moduli={0}.p_towers
)
{1}.control_gen(degree_layout=({0}.parameters.get('r', _default_r), {0}.parameters.get('c', _default_c)))
{1}.setup_relinearization(jnp.array({2}["a"], dtype=jnp.uint32).transpose(0,2,1), jnp.array({2}["b"], dtype=jnp.uint32).transpose(0,2,1))
)python";

// Template for GenRotKeyOp
// This template generates rotation keys for power-of-2 indices and initializes
// HERot. It computes r and c from the degree if they are not provided in the
// parameters.
constexpr std::string_view kGenRotKeyTemplate = R"python(
{0} = {{}}
_all_indices = [{1}]
_max_abs_rot_idx = max([abs(idx) for idx in _all_indices]) if _all_indices else 1
_power_of_2_indices = []
_pow2 = 1
while _pow2 <= _max_abs_rot_idx:
  _power_of_2_indices.append(_pow2)
  _pow2 <<= 1
_neg_power_of_2_indices = [-idx for idx in _power_of_2_indices]
_all_pow2_indices = _power_of_2_indices + _neg_power_of_2_indices

for _rot_idx in _all_pow2_indices:
  {0}[_rot_idx] = key_gen.gen_rotation_key({2}, {3}.q_towers, {3}.p_towers, rot_index=_rot_idx, dnum={3}.parameters.get('dnum', 3))

_degree_rot = {3}.parameters.get('degree')
if _degree_rot is not None:
  _log_degree_rot = int(math.log2(_degree_rot))
  _default_r_rot = 1 << (_log_degree_rot // 2)
  _default_c_rot = _degree_rot // _default_r_rot
else:
  _default_r_rot = 4
  _default_c_rot = 4
{4} = HERot(
    r={3}.parameters.get('r', _default_r_rot),
    c={3}.parameters.get('c', _default_c_rot),
    dnum={3}.parameters.get('dnum', 3),
    rotate_in_ciphertext_moduli={3}.q_towers,
    extend_moduli={3}.p_towers
)
{4}.control_gen(batch=1, degree_layout=({3}.parameters.get('r', _default_r_rot), {3}.parameters.get('c', _default_c_rot)))
{5} = {0}
)python";

// Template for DecryptOp
// This template prepares the ciphertext for decryption by extracting the
// required moduli.
constexpr std::string_view kDecryptTemplate = R"python(
{0}.secret_key = {1}
_rescales = {2}
_num_moduli = len({0}.q_towers) - _rescales * {0}.composite_degree
_q_sub = {0}.q_towers[:_num_moduli]
_ct_for_dec = Polynomial(
    {{'batch': 1, 'num_elements': 2, 'degree': {0}.degree,
     'precision': 32, 'num_moduli': _num_moduli,
     'degree_layout': ({0}.degree,)}},
    {{'moduli': _q_sub}})
_ct_for_dec.set_batch_polynomial({3}.polynomial.reshape(1, 2, {0}.degree, _num_moduli))
)python";

// Template for AddOp and AddInPlaceOp
// This template performs addition and modular reduction.
constexpr std::string_view kAddCoreTemplate = R"python(
{0}.add({1})
{0}.ciphertext = jnp.where({0}.ciphertext >= {0}.moduli_array, {0}.ciphertext - {0}.moduli_array, {0}.ciphertext)
)python";

// Template for SubOp, SubInPlaceOp, and SubPlainOp
// This template performs subtraction and modular reduction.
// {1} should be rhs.ciphertext for Sub/SubInPlace and just rhs for SubPlain.
constexpr std::string_view kSubTemplate = R"python(
{0}.ciphertext = jnp.where({0}.ciphertext < {1}, {0}.ciphertext + {0}.moduli_array - {1}, {0}.ciphertext - {1})
)python";

// Template for AddPlainOp modular reduction
constexpr std::string_view kAddModReduceTemplate = R"python(
{0}.ciphertext = jnp.where({0}.ciphertext >= {0}.moduli_array, {0}.ciphertext - {0}.moduli_array, {0}.ciphertext)
)python";

// Template for RelinOp
constexpr std::string_view kRelinTemplate = R"python(
{0} = {1}.he_mul[{2}].relinearize({3})
_s = {0}.polynomial.shape
{0}.polynomial = {0}.polynomial.reshape(_s[0], _s[1], {0}.degree, _s[-1])
)python";

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_JAXITEWORD_JAXITEWORDTEMPLATES_H_
