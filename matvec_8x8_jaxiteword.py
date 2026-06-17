import jax
import jax.numpy as jnp
import numpy as np
from ciphertext import Ciphertext
from polynomial import Polynomial
import ckks_ctx as ckks


def _assign_layout_15335824159471298539(
    v0: np.ndarray,
) -> np.ndarray:
  v1 = 8
  v2 = np.full(
      (
          8,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v3 = 0
  v4 = 1
  v5 = v2.copy()
  for v6 in range(0, 8):
    for v9 in range(0, 8):
      v11 = v6 + v9
      v12 = v11 % v1
      v13 = int(v9)
      v14 = int(v12)
      v15 = v0[v13, v14]
      v16 = int(v6)
      v5[v16, v13] = v15
  return v5


def matvec_identity__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> (
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
):
  v2 = np.array(
      [
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
      ],
      dtype=np.float32,
  ).reshape(8, 8)
  v3 = _assign_layout_15335824159471298539(v2)
  v4 = v3[3 : 3 + 1, 0 : 0 + 5]
  v5 = v3[3 : 3 + 1, 5 : 5 + 3]
  v6 = np.zeros(
      (
          1,
          8,
      ),
      dtype=np.float32,
  )
  v7 = v6.copy()
  v7[0 : 0 + 1, 3 : 3 + 5] = v4
  v8 = v7.copy()
  v8[0 : 0 + 1, 0 : 0 + 3] = v5
  v9 = v3[4 : 4 + 1, 0 : 0 + 5]
  v10 = v3[4 : 4 + 1, 5 : 5 + 3]
  v11 = v6.copy()
  v11[0 : 0 + 1, 3 : 3 + 5] = v9
  v12 = v11.copy()
  v12[0 : 0 + 1, 0 : 0 + 3] = v10
  v13 = v3[5 : 5 + 1, 0 : 0 + 5]
  v14 = v3[5 : 5 + 1, 5 : 5 + 3]
  v15 = v6.copy()
  v15[0 : 0 + 1, 3 : 3 + 5] = v13
  v16 = v15.copy()
  v16[0 : 0 + 1, 0 : 0 + 3] = v14
  v17 = v3[6 : 6 + 1, 0 : 0 + 2]
  v18 = v3[6 : 6 + 1, 2 : 2 + 6]
  v19 = v6.copy()
  v19[0 : 0 + 1, 6 : 6 + 2] = v17
  v20 = v19.copy()
  v20[0 : 0 + 1, 0 : 0 + 6] = v18
  v21 = v3[7 : 7 + 1, 0 : 0 + 2]
  v22 = v3[7 : 7 + 1, 2 : 2 + 6]
  v23 = v6.copy()
  v23[0 : 0 + 1, 6 : 6 + 2] = v21
  v24 = v23.copy()
  v24[0 : 0 + 1, 0 : 0 + 6] = v22
  v25 = v3[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v25)
  v26 = v3[1 : 1 + 1, 0 : 0 + 8].reshape(8)
  pt1 = v0.encode(v26)
  v27 = v3[2 : 2 + 1, 0 : 0 + 8].reshape(8)
  pt2 = v0.encode(v27)
  v28 = v8[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt3 = v0.encode(v28)
  v29 = v12[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt4 = v0.encode(v29)
  v30 = v16[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt5 = v0.encode(v30)
  v31 = v20[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt6 = v0.encode(v31)
  v32 = v24[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt7 = v0.encode(v32)
  v33 = [pt]
  v34 = [pt1]
  v35 = [pt2]
  v36 = [pt3]
  v37 = [pt4]
  v38 = [pt5]
  v39 = [pt6]
  v40 = [pt7]
  return (v33, v34, v35, v36, v37, v38, v39, v40)


def matvec_identity__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
    v4: np.ndarray,
    v5: np.ndarray,
    v6: np.ndarray,
    v7: np.ndarray,
    v8: np.ndarray,
    v9: np.ndarray,
    v10: np.ndarray,
) -> np.ndarray:
  v11 = 1
  v12 = 2
  v13 = 3
  v14 = 6
  v15 = 0
  pt = v3[0]
  pt1 = v4[0]
  pt2 = v5[0]
  pt3 = v6[0]
  pt4 = v7[0]
  pt5 = v8[0]
  pt6 = v9[0]
  pt7 = v10[0]
  ct = v2[0]
  _ct1_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct1_arg_m_in = _ct1_arg_data.shape[-1]
  _ct1_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct1_arg_m_in
  )
  _ct1_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct1_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct1_arg_r)
  )
  _ct1_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct1_arg_moduli, (int, np.integer)):
    _ct1_arg_moduli = [int(_ct1_arg_moduli)]
  ct1_arg = Polynomial(
      {
          "batch": _ct1_arg_data.shape[0],
          "num_elements": _ct1_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct1_arg_m,
          "precision": 32,
          "degree_layout": (_ct1_arg_r, _ct1_arg_c),
      },
      {"moduli": list(_ct1_arg_moduli)[:_ct1_arg_m]},
  )
  ct1_arg.polynomial = _ct1_arg_data.reshape(
      _ct1_arg_data.shape[0],
      _ct1_arg_data.shape[1],
      _ct1_arg_r,
      _ct1_arg_c,
      _ct1_arg_m_in,
  )[..., :_ct1_arg_m].copy()
  ct1_arg.batch = ct1_arg.polynomial.shape[0]
  ct1_arg.num_elements = ct1_arg.polynomial.shape[1]
  ct1_arg.num_moduli = _ct1_arg_m
  ct1_arg.degree_layout = (_ct1_arg_r, _ct1_arg_c)
  ct1_arg.r = _ct1_arg_r
  ct1_arg.c = _ct1_arg_c
  ct1_arg.moduli = list(_ct1_arg_moduli)[:_ct1_arg_m]
  ct1_arg.moduli_array = jnp.array(
      ct1_arg.moduli, dtype=getattr(ct1_arg, "modulus_dtype", jnp.uint32)
  )
  ct1_pt_ntt = (
      pt.polynomial[0, 0, :, : ct1_arg.polynomial.shape[-1]]
      .reshape(ct1_arg.r, ct1_arg.c, ct1_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct1_ptct = v0.ptct_mul[v0.max_level]
  ct1_ptct.set_plaintext(ct1_pt_ntt)
  ct1_raw = ct1_ptct.mul(ct1_arg, use_bat=False)
  _ct1_data = ct1_raw.polynomial if hasattr(ct1_raw, "polynomial") else ct1_raw
  _ct1_m_in = _ct1_data.shape[-1]
  _ct1_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct1_m_in
  )
  _ct1_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct1_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct1_r)
  )
  _ct1_moduli = getattr(ct1_raw, "moduli", v0.q_towers)
  if isinstance(_ct1_moduli, (int, np.integer)):
    _ct1_moduli = [int(_ct1_moduli)]
  ct1 = Polynomial(
      {
          "batch": _ct1_data.shape[0],
          "num_elements": _ct1_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct1_m,
          "precision": 32,
          "degree_layout": (_ct1_r, _ct1_c),
      },
      {"moduli": list(_ct1_moduli)[:_ct1_m]},
  )
  ct1.polynomial = _ct1_data.reshape(
      _ct1_data.shape[0], _ct1_data.shape[1], _ct1_r, _ct1_c, _ct1_m_in
  )[..., :_ct1_m].copy()
  ct1.batch = ct1.polynomial.shape[0]
  ct1.num_elements = ct1.polynomial.shape[1]
  ct1.num_moduli = _ct1_m
  ct1.degree_layout = (_ct1_r, _ct1_c)
  ct1.r = _ct1_r
  ct1.c = _ct1_c
  ct1.moduli = list(_ct1_moduli)[:_ct1_m]
  ct1.moduli_array = jnp.array(
      ct1.moduli, dtype=getattr(ct1, "modulus_dtype", jnp.uint32)
  )
  _ct2_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct2_arg_m_in = _ct2_arg_data.shape[-1]
  _ct2_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct2_arg_m_in
  )
  _ct2_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct2_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct2_arg_r)
  )
  _ct2_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct2_arg_moduli, (int, np.integer)):
    _ct2_arg_moduli = [int(_ct2_arg_moduli)]
  ct2_arg = Polynomial(
      {
          "batch": _ct2_arg_data.shape[0],
          "num_elements": _ct2_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct2_arg_m,
          "precision": 32,
          "degree_layout": (_ct2_arg_r, _ct2_arg_c),
      },
      {"moduli": list(_ct2_arg_moduli)[:_ct2_arg_m]},
  )
  ct2_arg.polynomial = _ct2_arg_data.reshape(
      _ct2_arg_data.shape[0],
      _ct2_arg_data.shape[1],
      _ct2_arg_r,
      _ct2_arg_c,
      _ct2_arg_m_in,
  )[..., :_ct2_arg_m].copy()
  ct2_arg.batch = ct2_arg.polynomial.shape[0]
  ct2_arg.num_elements = ct2_arg.polynomial.shape[1]
  ct2_arg.num_moduli = _ct2_arg_m
  ct2_arg.degree_layout = (_ct2_arg_r, _ct2_arg_c)
  ct2_arg.r = _ct2_arg_r
  ct2_arg.c = _ct2_arg_c
  ct2_arg.moduli = list(_ct2_arg_moduli)[:_ct2_arg_m]
  ct2_arg.moduli_array = jnp.array(
      ct2_arg.moduli, dtype=getattr(ct2_arg, "modulus_dtype", jnp.uint32)
  )
  ct2_raw = v0.he_rot[v0.max_level, 1].rotate(ct2_arg)
  _ct2_data = ct2_raw.polynomial if hasattr(ct2_raw, "polynomial") else ct2_raw
  _ct2_m_in = _ct2_data.shape[-1]
  _ct2_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct2_m_in
  )
  _ct2_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct2_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct2_r)
  )
  _ct2_moduli = getattr(ct2_raw, "moduli", v0.q_towers)
  if isinstance(_ct2_moduli, (int, np.integer)):
    _ct2_moduli = [int(_ct2_moduli)]
  ct2 = Polynomial(
      {
          "batch": _ct2_data.shape[0],
          "num_elements": _ct2_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct2_m,
          "precision": 32,
          "degree_layout": (_ct2_r, _ct2_c),
      },
      {"moduli": list(_ct2_moduli)[:_ct2_m]},
  )
  ct2.polynomial = _ct2_data.reshape(
      _ct2_data.shape[0], _ct2_data.shape[1], _ct2_r, _ct2_c, _ct2_m_in
  )[..., :_ct2_m].copy()
  ct2.batch = ct2.polynomial.shape[0]
  ct2.num_elements = ct2.polynomial.shape[1]
  ct2.num_moduli = _ct2_m
  ct2.degree_layout = (_ct2_r, _ct2_c)
  ct2.r = _ct2_r
  ct2.c = _ct2_c
  ct2.moduli = list(_ct2_moduli)[:_ct2_m]
  ct2.moduli_array = jnp.array(
      ct2.moduli, dtype=getattr(ct2, "modulus_dtype", jnp.uint32)
  )
  _ct3_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct3_arg_m_in = _ct3_arg_data.shape[-1]
  _ct3_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct3_arg_m_in
  )
  _ct3_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct3_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct3_arg_r)
  )
  _ct3_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct3_arg_moduli, (int, np.integer)):
    _ct3_arg_moduli = [int(_ct3_arg_moduli)]
  ct3_arg = Polynomial(
      {
          "batch": _ct3_arg_data.shape[0],
          "num_elements": _ct3_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct3_arg_m,
          "precision": 32,
          "degree_layout": (_ct3_arg_r, _ct3_arg_c),
      },
      {"moduli": list(_ct3_arg_moduli)[:_ct3_arg_m]},
  )
  ct3_arg.polynomial = _ct3_arg_data.reshape(
      _ct3_arg_data.shape[0],
      _ct3_arg_data.shape[1],
      _ct3_arg_r,
      _ct3_arg_c,
      _ct3_arg_m_in,
  )[..., :_ct3_arg_m].copy()
  ct3_arg.batch = ct3_arg.polynomial.shape[0]
  ct3_arg.num_elements = ct3_arg.polynomial.shape[1]
  ct3_arg.num_moduli = _ct3_arg_m
  ct3_arg.degree_layout = (_ct3_arg_r, _ct3_arg_c)
  ct3_arg.r = _ct3_arg_r
  ct3_arg.c = _ct3_arg_c
  ct3_arg.moduli = list(_ct3_arg_moduli)[:_ct3_arg_m]
  ct3_arg.moduli_array = jnp.array(
      ct3_arg.moduli, dtype=getattr(ct3_arg, "modulus_dtype", jnp.uint32)
  )
  ct3_pt_ntt = (
      pt1.polynomial[0, 0, :, : ct3_arg.polynomial.shape[-1]]
      .reshape(ct3_arg.r, ct3_arg.c, ct3_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct3_ptct = v0.ptct_mul[v0.max_level]
  ct3_ptct.set_plaintext(ct3_pt_ntt)
  ct3_raw = ct3_ptct.mul(ct3_arg, use_bat=False)
  _ct3_data = ct3_raw.polynomial if hasattr(ct3_raw, "polynomial") else ct3_raw
  _ct3_m_in = _ct3_data.shape[-1]
  _ct3_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct3_m_in
  )
  _ct3_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct3_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct3_r)
  )
  _ct3_moduli = getattr(ct3_raw, "moduli", v0.q_towers)
  if isinstance(_ct3_moduli, (int, np.integer)):
    _ct3_moduli = [int(_ct3_moduli)]
  ct3 = Polynomial(
      {
          "batch": _ct3_data.shape[0],
          "num_elements": _ct3_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct3_m,
          "precision": 32,
          "degree_layout": (_ct3_r, _ct3_c),
      },
      {"moduli": list(_ct3_moduli)[:_ct3_m]},
  )
  ct3.polynomial = _ct3_data.reshape(
      _ct3_data.shape[0], _ct3_data.shape[1], _ct3_r, _ct3_c, _ct3_m_in
  )[..., :_ct3_m].copy()
  ct3.batch = ct3.polynomial.shape[0]
  ct3.num_elements = ct3.polynomial.shape[1]
  ct3.num_moduli = _ct3_m
  ct3.degree_layout = (_ct3_r, _ct3_c)
  ct3.r = _ct3_r
  ct3.c = _ct3_c
  ct3.moduli = list(_ct3_moduli)[:_ct3_m]
  ct3.moduli_array = jnp.array(
      ct3.moduli, dtype=getattr(ct3, "modulus_dtype", jnp.uint32)
  )
  _ct4_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct4_arg_m_in = _ct4_arg_data.shape[-1]
  _ct4_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct4_arg_m_in
  )
  _ct4_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct4_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct4_arg_r)
  )
  _ct4_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct4_arg_moduli, (int, np.integer)):
    _ct4_arg_moduli = [int(_ct4_arg_moduli)]
  ct4_arg = Polynomial(
      {
          "batch": _ct4_arg_data.shape[0],
          "num_elements": _ct4_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct4_arg_m,
          "precision": 32,
          "degree_layout": (_ct4_arg_r, _ct4_arg_c),
      },
      {"moduli": list(_ct4_arg_moduli)[:_ct4_arg_m]},
  )
  ct4_arg.polynomial = _ct4_arg_data.reshape(
      _ct4_arg_data.shape[0],
      _ct4_arg_data.shape[1],
      _ct4_arg_r,
      _ct4_arg_c,
      _ct4_arg_m_in,
  )[..., :_ct4_arg_m].copy()
  ct4_arg.batch = ct4_arg.polynomial.shape[0]
  ct4_arg.num_elements = ct4_arg.polynomial.shape[1]
  ct4_arg.num_moduli = _ct4_arg_m
  ct4_arg.degree_layout = (_ct4_arg_r, _ct4_arg_c)
  ct4_arg.r = _ct4_arg_r
  ct4_arg.c = _ct4_arg_c
  ct4_arg.moduli = list(_ct4_arg_moduli)[:_ct4_arg_m]
  ct4_arg.moduli_array = jnp.array(
      ct4_arg.moduli, dtype=getattr(ct4_arg, "modulus_dtype", jnp.uint32)
  )
  ct4_raw = v0.he_rot[v0.max_level, 2].rotate(ct4_arg)
  _ct4_data = ct4_raw.polynomial if hasattr(ct4_raw, "polynomial") else ct4_raw
  _ct4_m_in = _ct4_data.shape[-1]
  _ct4_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct4_m_in
  )
  _ct4_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct4_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct4_r)
  )
  _ct4_moduli = getattr(ct4_raw, "moduli", v0.q_towers)
  if isinstance(_ct4_moduli, (int, np.integer)):
    _ct4_moduli = [int(_ct4_moduli)]
  ct4 = Polynomial(
      {
          "batch": _ct4_data.shape[0],
          "num_elements": _ct4_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct4_m,
          "precision": 32,
          "degree_layout": (_ct4_r, _ct4_c),
      },
      {"moduli": list(_ct4_moduli)[:_ct4_m]},
  )
  ct4.polynomial = _ct4_data.reshape(
      _ct4_data.shape[0], _ct4_data.shape[1], _ct4_r, _ct4_c, _ct4_m_in
  )[..., :_ct4_m].copy()
  ct4.batch = ct4.polynomial.shape[0]
  ct4.num_elements = ct4.polynomial.shape[1]
  ct4.num_moduli = _ct4_m
  ct4.degree_layout = (_ct4_r, _ct4_c)
  ct4.r = _ct4_r
  ct4.c = _ct4_c
  ct4.moduli = list(_ct4_moduli)[:_ct4_m]
  ct4.moduli_array = jnp.array(
      ct4.moduli, dtype=getattr(ct4, "modulus_dtype", jnp.uint32)
  )
  _ct5_arg_data = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  _ct5_arg_m_in = _ct5_arg_data.shape[-1]
  _ct5_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct5_arg_m_in
  )
  _ct5_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct5_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct5_arg_r)
  )
  _ct5_arg_moduli = getattr(ct4, "moduli", v0.q_towers)
  if isinstance(_ct5_arg_moduli, (int, np.integer)):
    _ct5_arg_moduli = [int(_ct5_arg_moduli)]
  ct5_arg = Polynomial(
      {
          "batch": _ct5_arg_data.shape[0],
          "num_elements": _ct5_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct5_arg_m,
          "precision": 32,
          "degree_layout": (_ct5_arg_r, _ct5_arg_c),
      },
      {"moduli": list(_ct5_arg_moduli)[:_ct5_arg_m]},
  )
  ct5_arg.polynomial = _ct5_arg_data.reshape(
      _ct5_arg_data.shape[0],
      _ct5_arg_data.shape[1],
      _ct5_arg_r,
      _ct5_arg_c,
      _ct5_arg_m_in,
  )[..., :_ct5_arg_m].copy()
  ct5_arg.batch = ct5_arg.polynomial.shape[0]
  ct5_arg.num_elements = ct5_arg.polynomial.shape[1]
  ct5_arg.num_moduli = _ct5_arg_m
  ct5_arg.degree_layout = (_ct5_arg_r, _ct5_arg_c)
  ct5_arg.r = _ct5_arg_r
  ct5_arg.c = _ct5_arg_c
  ct5_arg.moduli = list(_ct5_arg_moduli)[:_ct5_arg_m]
  ct5_arg.moduli_array = jnp.array(
      ct5_arg.moduli, dtype=getattr(ct5_arg, "modulus_dtype", jnp.uint32)
  )
  ct5_pt_ntt = (
      pt2.polynomial[0, 0, :, : ct5_arg.polynomial.shape[-1]]
      .reshape(ct5_arg.r, ct5_arg.c, ct5_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5_raw = ct5_ptct.mul(ct5_arg, use_bat=False)
  _ct5_data = ct5_raw.polynomial if hasattr(ct5_raw, "polynomial") else ct5_raw
  _ct5_m_in = _ct5_data.shape[-1]
  _ct5_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct5_m_in
  )
  _ct5_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct5_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct5_r)
  )
  _ct5_moduli = getattr(ct5_raw, "moduli", v0.q_towers)
  if isinstance(_ct5_moduli, (int, np.integer)):
    _ct5_moduli = [int(_ct5_moduli)]
  ct5 = Polynomial(
      {
          "batch": _ct5_data.shape[0],
          "num_elements": _ct5_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct5_m,
          "precision": 32,
          "degree_layout": (_ct5_r, _ct5_c),
      },
      {"moduli": list(_ct5_moduli)[:_ct5_m]},
  )
  ct5.polynomial = _ct5_data.reshape(
      _ct5_data.shape[0], _ct5_data.shape[1], _ct5_r, _ct5_c, _ct5_m_in
  )[..., :_ct5_m].copy()
  ct5.batch = ct5.polynomial.shape[0]
  ct5.num_elements = ct5.polynomial.shape[1]
  ct5.num_moduli = _ct5_m
  ct5.degree_layout = (_ct5_r, _ct5_c)
  ct5.r = _ct5_r
  ct5.c = _ct5_c
  ct5.moduli = list(_ct5_moduli)[:_ct5_m]
  ct5.moduli_array = jnp.array(
      ct5.moduli, dtype=getattr(ct5, "modulus_dtype", jnp.uint32)
  )
  _ct6_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct6_arg_m_in = _ct6_arg_data.shape[-1]
  _ct6_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct6_arg_m_in
  )
  _ct6_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct6_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct6_arg_r)
  )
  _ct6_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct6_arg_moduli, (int, np.integer)):
    _ct6_arg_moduli = [int(_ct6_arg_moduli)]
  ct6_arg = Polynomial(
      {
          "batch": _ct6_arg_data.shape[0],
          "num_elements": _ct6_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct6_arg_m,
          "precision": 32,
          "degree_layout": (_ct6_arg_r, _ct6_arg_c),
      },
      {"moduli": list(_ct6_arg_moduli)[:_ct6_arg_m]},
  )
  ct6_arg.polynomial = _ct6_arg_data.reshape(
      _ct6_arg_data.shape[0],
      _ct6_arg_data.shape[1],
      _ct6_arg_r,
      _ct6_arg_c,
      _ct6_arg_m_in,
  )[..., :_ct6_arg_m].copy()
  ct6_arg.batch = ct6_arg.polynomial.shape[0]
  ct6_arg.num_elements = ct6_arg.polynomial.shape[1]
  ct6_arg.num_moduli = _ct6_arg_m
  ct6_arg.degree_layout = (_ct6_arg_r, _ct6_arg_c)
  ct6_arg.r = _ct6_arg_r
  ct6_arg.c = _ct6_arg_c
  ct6_arg.moduli = list(_ct6_arg_moduli)[:_ct6_arg_m]
  ct6_arg.moduli_array = jnp.array(
      ct6_arg.moduli, dtype=getattr(ct6_arg, "modulus_dtype", jnp.uint32)
  )
  ct6_pt_ntt = (
      pt3.polynomial[0, 0, :, : ct6_arg.polynomial.shape[-1]]
      .reshape(ct6_arg.r, ct6_arg.c, ct6_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct6_ptct = v0.ptct_mul[v0.max_level]
  ct6_ptct.set_plaintext(ct6_pt_ntt)
  ct6_raw = ct6_ptct.mul(ct6_arg, use_bat=False)
  _ct6_data = ct6_raw.polynomial if hasattr(ct6_raw, "polynomial") else ct6_raw
  _ct6_m_in = _ct6_data.shape[-1]
  _ct6_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct6_m_in
  )
  _ct6_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct6_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct6_r)
  )
  _ct6_moduli = getattr(ct6_raw, "moduli", v0.q_towers)
  if isinstance(_ct6_moduli, (int, np.integer)):
    _ct6_moduli = [int(_ct6_moduli)]
  ct6 = Polynomial(
      {
          "batch": _ct6_data.shape[0],
          "num_elements": _ct6_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct6_m,
          "precision": 32,
          "degree_layout": (_ct6_r, _ct6_c),
      },
      {"moduli": list(_ct6_moduli)[:_ct6_m]},
  )
  ct6.polynomial = _ct6_data.reshape(
      _ct6_data.shape[0], _ct6_data.shape[1], _ct6_r, _ct6_c, _ct6_m_in
  )[..., :_ct6_m].copy()
  ct6.batch = ct6.polynomial.shape[0]
  ct6.num_elements = ct6.polynomial.shape[1]
  ct6.num_moduli = _ct6_m
  ct6.degree_layout = (_ct6_r, _ct6_c)
  ct6.r = _ct6_r
  ct6.c = _ct6_c
  ct6.moduli = list(_ct6_moduli)[:_ct6_m]
  ct6.moduli_array = jnp.array(
      ct6.moduli, dtype=getattr(ct6, "modulus_dtype", jnp.uint32)
  )
  _ct7_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct7_arg_m_in = _ct7_arg_data.shape[-1]
  _ct7_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct7_arg_m_in
  )
  _ct7_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct7_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct7_arg_r)
  )
  _ct7_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct7_arg_moduli, (int, np.integer)):
    _ct7_arg_moduli = [int(_ct7_arg_moduli)]
  ct7_arg = Polynomial(
      {
          "batch": _ct7_arg_data.shape[0],
          "num_elements": _ct7_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct7_arg_m,
          "precision": 32,
          "degree_layout": (_ct7_arg_r, _ct7_arg_c),
      },
      {"moduli": list(_ct7_arg_moduli)[:_ct7_arg_m]},
  )
  ct7_arg.polynomial = _ct7_arg_data.reshape(
      _ct7_arg_data.shape[0],
      _ct7_arg_data.shape[1],
      _ct7_arg_r,
      _ct7_arg_c,
      _ct7_arg_m_in,
  )[..., :_ct7_arg_m].copy()
  ct7_arg.batch = ct7_arg.polynomial.shape[0]
  ct7_arg.num_elements = ct7_arg.polynomial.shape[1]
  ct7_arg.num_moduli = _ct7_arg_m
  ct7_arg.degree_layout = (_ct7_arg_r, _ct7_arg_c)
  ct7_arg.r = _ct7_arg_r
  ct7_arg.c = _ct7_arg_c
  ct7_arg.moduli = list(_ct7_arg_moduli)[:_ct7_arg_m]
  ct7_arg.moduli_array = jnp.array(
      ct7_arg.moduli, dtype=getattr(ct7_arg, "modulus_dtype", jnp.uint32)
  )
  ct7_pt_ntt = (
      pt4.polynomial[0, 0, :, : ct7_arg.polynomial.shape[-1]]
      .reshape(ct7_arg.r, ct7_arg.c, ct7_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct7_ptct = v0.ptct_mul[v0.max_level]
  ct7_ptct.set_plaintext(ct7_pt_ntt)
  ct7_raw = ct7_ptct.mul(ct7_arg, use_bat=False)
  _ct7_data = ct7_raw.polynomial if hasattr(ct7_raw, "polynomial") else ct7_raw
  _ct7_m_in = _ct7_data.shape[-1]
  _ct7_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct7_m_in
  )
  _ct7_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct7_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct7_r)
  )
  _ct7_moduli = getattr(ct7_raw, "moduli", v0.q_towers)
  if isinstance(_ct7_moduli, (int, np.integer)):
    _ct7_moduli = [int(_ct7_moduli)]
  ct7 = Polynomial(
      {
          "batch": _ct7_data.shape[0],
          "num_elements": _ct7_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct7_m,
          "precision": 32,
          "degree_layout": (_ct7_r, _ct7_c),
      },
      {"moduli": list(_ct7_moduli)[:_ct7_m]},
  )
  ct7.polynomial = _ct7_data.reshape(
      _ct7_data.shape[0], _ct7_data.shape[1], _ct7_r, _ct7_c, _ct7_m_in
  )[..., :_ct7_m].copy()
  ct7.batch = ct7.polynomial.shape[0]
  ct7.num_elements = ct7.polynomial.shape[1]
  ct7.num_moduli = _ct7_m
  ct7.degree_layout = (_ct7_r, _ct7_c)
  ct7.r = _ct7_r
  ct7.c = _ct7_c
  ct7.moduli = list(_ct7_moduli)[:_ct7_m]
  ct7.moduli_array = jnp.array(
      ct7.moduli, dtype=getattr(ct7, "modulus_dtype", jnp.uint32)
  )
  _ct8_arg_data = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  _ct8_arg_m_in = _ct8_arg_data.shape[-1]
  _ct8_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct8_arg_m_in
  )
  _ct8_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct8_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct8_arg_r)
  )
  _ct8_arg_moduli = getattr(ct4, "moduli", v0.q_towers)
  if isinstance(_ct8_arg_moduli, (int, np.integer)):
    _ct8_arg_moduli = [int(_ct8_arg_moduli)]
  ct8_arg = Polynomial(
      {
          "batch": _ct8_arg_data.shape[0],
          "num_elements": _ct8_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct8_arg_m,
          "precision": 32,
          "degree_layout": (_ct8_arg_r, _ct8_arg_c),
      },
      {"moduli": list(_ct8_arg_moduli)[:_ct8_arg_m]},
  )
  ct8_arg.polynomial = _ct8_arg_data.reshape(
      _ct8_arg_data.shape[0],
      _ct8_arg_data.shape[1],
      _ct8_arg_r,
      _ct8_arg_c,
      _ct8_arg_m_in,
  )[..., :_ct8_arg_m].copy()
  ct8_arg.batch = ct8_arg.polynomial.shape[0]
  ct8_arg.num_elements = ct8_arg.polynomial.shape[1]
  ct8_arg.num_moduli = _ct8_arg_m
  ct8_arg.degree_layout = (_ct8_arg_r, _ct8_arg_c)
  ct8_arg.r = _ct8_arg_r
  ct8_arg.c = _ct8_arg_c
  ct8_arg.moduli = list(_ct8_arg_moduli)[:_ct8_arg_m]
  ct8_arg.moduli_array = jnp.array(
      ct8_arg.moduli, dtype=getattr(ct8_arg, "modulus_dtype", jnp.uint32)
  )
  ct8_pt_ntt = (
      pt5.polynomial[0, 0, :, : ct8_arg.polynomial.shape[-1]]
      .reshape(ct8_arg.r, ct8_arg.c, ct8_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct8_ptct = v0.ptct_mul[v0.max_level]
  ct8_ptct.set_plaintext(ct8_pt_ntt)
  ct8_raw = ct8_ptct.mul(ct8_arg, use_bat=False)
  _ct8_data = ct8_raw.polynomial if hasattr(ct8_raw, "polynomial") else ct8_raw
  _ct8_m_in = _ct8_data.shape[-1]
  _ct8_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct8_m_in
  )
  _ct8_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct8_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct8_r)
  )
  _ct8_moduli = getattr(ct8_raw, "moduli", v0.q_towers)
  if isinstance(_ct8_moduli, (int, np.integer)):
    _ct8_moduli = [int(_ct8_moduli)]
  ct8 = Polynomial(
      {
          "batch": _ct8_data.shape[0],
          "num_elements": _ct8_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct8_m,
          "precision": 32,
          "degree_layout": (_ct8_r, _ct8_c),
      },
      {"moduli": list(_ct8_moduli)[:_ct8_m]},
  )
  ct8.polynomial = _ct8_data.reshape(
      _ct8_data.shape[0], _ct8_data.shape[1], _ct8_r, _ct8_c, _ct8_m_in
  )[..., :_ct8_m].copy()
  ct8.batch = ct8.polynomial.shape[0]
  ct8.num_elements = ct8.polynomial.shape[1]
  ct8.num_moduli = _ct8_m
  ct8.degree_layout = (_ct8_r, _ct8_c)
  ct8.r = _ct8_r
  ct8.c = _ct8_c
  ct8.moduli = list(_ct8_moduli)[:_ct8_m]
  ct8.moduli_array = jnp.array(
      ct8.moduli, dtype=getattr(ct8, "modulus_dtype", jnp.uint32)
  )
  _ct9_data = ct6.polynomial if hasattr(ct6, "polynomial") else ct6
  _ct9_m_in = _ct9_data.shape[-1]
  _ct9_m = _ct9_m_in
  _ct9_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct9_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct9_r)
  )
  _ct9_moduli = getattr(ct6, "moduli", v0.q_towers)
  if isinstance(_ct9_moduli, (int, np.integer)):
    _ct9_moduli = [int(_ct9_moduli)]
  ct9 = Polynomial(
      {
          "batch": _ct9_data.shape[0],
          "num_elements": _ct9_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct9_m,
          "precision": 32,
          "degree_layout": (_ct9_r, _ct9_c),
      },
      {"moduli": list(_ct9_moduli)[:_ct9_m]},
  )
  ct9.polynomial = _ct9_data.reshape(
      _ct9_data.shape[0], _ct9_data.shape[1], _ct9_r, _ct9_c, _ct9_m_in
  )[..., :_ct9_m].copy()
  ct9.batch = ct9.polynomial.shape[0]
  ct9.num_elements = ct9.polynomial.shape[1]
  ct9.num_moduli = _ct9_m
  ct9.degree_layout = (_ct9_r, _ct9_c)
  ct9.r = _ct9_r
  ct9.c = _ct9_c
  ct9.moduli = list(_ct9_moduli)[:_ct9_m]
  ct9.moduli_array = jnp.array(
      ct9.moduli, dtype=getattr(ct9, "modulus_dtype", jnp.uint32)
  )
  _ct9_rhs_data = ct7.polynomial if hasattr(ct7, "polynomial") else ct7
  _ct9_rhs_m_in = _ct9_rhs_data.shape[-1]
  _ct9_rhs_m = _ct9_rhs_m_in
  _ct9_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct9_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct9_rhs_r)
  )
  _ct9_rhs_moduli = getattr(ct7, "moduli", v0.q_towers)
  if isinstance(_ct9_rhs_moduli, (int, np.integer)):
    _ct9_rhs_moduli = [int(_ct9_rhs_moduli)]
  ct9_rhs = Polynomial(
      {
          "batch": _ct9_rhs_data.shape[0],
          "num_elements": _ct9_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct9_rhs_m,
          "precision": 32,
          "degree_layout": (_ct9_rhs_r, _ct9_rhs_c),
      },
      {"moduli": list(_ct9_rhs_moduli)[:_ct9_rhs_m]},
  )
  ct9_rhs.polynomial = _ct9_rhs_data.reshape(
      _ct9_rhs_data.shape[0],
      _ct9_rhs_data.shape[1],
      _ct9_rhs_r,
      _ct9_rhs_c,
      _ct9_rhs_m_in,
  )[..., :_ct9_rhs_m].copy()
  ct9_rhs.batch = ct9_rhs.polynomial.shape[0]
  ct9_rhs.num_elements = ct9_rhs.polynomial.shape[1]
  ct9_rhs.num_moduli = _ct9_rhs_m
  ct9_rhs.degree_layout = (_ct9_rhs_r, _ct9_rhs_c)
  ct9_rhs.r = _ct9_rhs_r
  ct9_rhs.c = _ct9_rhs_c
  ct9_rhs.moduli = list(_ct9_rhs_moduli)[:_ct9_rhs_m]
  ct9_rhs.moduli_array = jnp.array(
      ct9_rhs.moduli, dtype=getattr(ct9_rhs, "modulus_dtype", jnp.uint32)
  )
  ct9.add(ct9_rhs)
  _moduli = jnp.array(ct9.moduli, dtype=jnp.uint32)
  ct9.polynomial = jnp.where(
      ct9.polynomial >= _moduli, ct9.polynomial - _moduli, ct9.polynomial
  )
  _ct10_data = ct9.polynomial if hasattr(ct9, "polynomial") else ct9
  _ct10_m_in = _ct10_data.shape[-1]
  _ct10_m = _ct10_m_in
  _ct10_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct10_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct10_r)
  )
  _ct10_moduli = getattr(ct9, "moduli", v0.q_towers)
  if isinstance(_ct10_moduli, (int, np.integer)):
    _ct10_moduli = [int(_ct10_moduli)]
  ct10 = Polynomial(
      {
          "batch": _ct10_data.shape[0],
          "num_elements": _ct10_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct10_m,
          "precision": 32,
          "degree_layout": (_ct10_r, _ct10_c),
      },
      {"moduli": list(_ct10_moduli)[:_ct10_m]},
  )
  ct10.polynomial = _ct10_data.reshape(
      _ct10_data.shape[0], _ct10_data.shape[1], _ct10_r, _ct10_c, _ct10_m_in
  )[..., :_ct10_m].copy()
  ct10.batch = ct10.polynomial.shape[0]
  ct10.num_elements = ct10.polynomial.shape[1]
  ct10.num_moduli = _ct10_m
  ct10.degree_layout = (_ct10_r, _ct10_c)
  ct10.r = _ct10_r
  ct10.c = _ct10_c
  ct10.moduli = list(_ct10_moduli)[:_ct10_m]
  ct10.moduli_array = jnp.array(
      ct10.moduli, dtype=getattr(ct10, "modulus_dtype", jnp.uint32)
  )
  _ct10_rhs_data = ct8.polynomial if hasattr(ct8, "polynomial") else ct8
  _ct10_rhs_m_in = _ct10_rhs_data.shape[-1]
  _ct10_rhs_m = _ct10_rhs_m_in
  _ct10_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct10_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct10_rhs_r)
  )
  _ct10_rhs_moduli = getattr(ct8, "moduli", v0.q_towers)
  if isinstance(_ct10_rhs_moduli, (int, np.integer)):
    _ct10_rhs_moduli = [int(_ct10_rhs_moduli)]
  ct10_rhs = Polynomial(
      {
          "batch": _ct10_rhs_data.shape[0],
          "num_elements": _ct10_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct10_rhs_m,
          "precision": 32,
          "degree_layout": (_ct10_rhs_r, _ct10_rhs_c),
      },
      {"moduli": list(_ct10_rhs_moduli)[:_ct10_rhs_m]},
  )
  ct10_rhs.polynomial = _ct10_rhs_data.reshape(
      _ct10_rhs_data.shape[0],
      _ct10_rhs_data.shape[1],
      _ct10_rhs_r,
      _ct10_rhs_c,
      _ct10_rhs_m_in,
  )[..., :_ct10_rhs_m].copy()
  ct10_rhs.batch = ct10_rhs.polynomial.shape[0]
  ct10_rhs.num_elements = ct10_rhs.polynomial.shape[1]
  ct10_rhs.num_moduli = _ct10_rhs_m
  ct10_rhs.degree_layout = (_ct10_rhs_r, _ct10_rhs_c)
  ct10_rhs.r = _ct10_rhs_r
  ct10_rhs.c = _ct10_rhs_c
  ct10_rhs.moduli = list(_ct10_rhs_moduli)[:_ct10_rhs_m]
  ct10_rhs.moduli_array = jnp.array(
      ct10_rhs.moduli, dtype=getattr(ct10_rhs, "modulus_dtype", jnp.uint32)
  )
  ct10.add(ct10_rhs)
  _moduli = jnp.array(ct10.moduli, dtype=jnp.uint32)
  ct10.polynomial = jnp.where(
      ct10.polynomial >= _moduli, ct10.polynomial - _moduli, ct10.polynomial
  )
  _ct11_arg_data = ct10.polynomial if hasattr(ct10, "polynomial") else ct10
  _ct11_arg_m_in = _ct11_arg_data.shape[-1]
  _ct11_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct11_arg_m_in
  )
  _ct11_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct11_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct11_arg_r)
  )
  _ct11_arg_moduli = getattr(ct10, "moduli", v0.q_towers)
  if isinstance(_ct11_arg_moduli, (int, np.integer)):
    _ct11_arg_moduli = [int(_ct11_arg_moduli)]
  ct11_arg = Polynomial(
      {
          "batch": _ct11_arg_data.shape[0],
          "num_elements": _ct11_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct11_arg_m,
          "precision": 32,
          "degree_layout": (_ct11_arg_r, _ct11_arg_c),
      },
      {"moduli": list(_ct11_arg_moduli)[:_ct11_arg_m]},
  )
  ct11_arg.polynomial = _ct11_arg_data.reshape(
      _ct11_arg_data.shape[0],
      _ct11_arg_data.shape[1],
      _ct11_arg_r,
      _ct11_arg_c,
      _ct11_arg_m_in,
  )[..., :_ct11_arg_m].copy()
  ct11_arg.batch = ct11_arg.polynomial.shape[0]
  ct11_arg.num_elements = ct11_arg.polynomial.shape[1]
  ct11_arg.num_moduli = _ct11_arg_m
  ct11_arg.degree_layout = (_ct11_arg_r, _ct11_arg_c)
  ct11_arg.r = _ct11_arg_r
  ct11_arg.c = _ct11_arg_c
  ct11_arg.moduli = list(_ct11_arg_moduli)[:_ct11_arg_m]
  ct11_arg.moduli_array = jnp.array(
      ct11_arg.moduli, dtype=getattr(ct11_arg, "modulus_dtype", jnp.uint32)
  )
  ct11_raw = v0.he_rot[v0.max_level, 3].rotate(ct11_arg)
  _ct11_data = (
      ct11_raw.polynomial if hasattr(ct11_raw, "polynomial") else ct11_raw
  )
  _ct11_m_in = _ct11_data.shape[-1]
  _ct11_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct11_m_in
  )
  _ct11_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct11_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct11_r)
  )
  _ct11_moduli = getattr(ct11_raw, "moduli", v0.q_towers)
  if isinstance(_ct11_moduli, (int, np.integer)):
    _ct11_moduli = [int(_ct11_moduli)]
  ct11 = Polynomial(
      {
          "batch": _ct11_data.shape[0],
          "num_elements": _ct11_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct11_m,
          "precision": 32,
          "degree_layout": (_ct11_r, _ct11_c),
      },
      {"moduli": list(_ct11_moduli)[:_ct11_m]},
  )
  ct11.polynomial = _ct11_data.reshape(
      _ct11_data.shape[0], _ct11_data.shape[1], _ct11_r, _ct11_c, _ct11_m_in
  )[..., :_ct11_m].copy()
  ct11.batch = ct11.polynomial.shape[0]
  ct11.num_elements = ct11.polynomial.shape[1]
  ct11.num_moduli = _ct11_m
  ct11.degree_layout = (_ct11_r, _ct11_c)
  ct11.r = _ct11_r
  ct11.c = _ct11_c
  ct11.moduli = list(_ct11_moduli)[:_ct11_m]
  ct11.moduli_array = jnp.array(
      ct11.moduli, dtype=getattr(ct11, "modulus_dtype", jnp.uint32)
  )
  _ct12_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct12_arg_m_in = _ct12_arg_data.shape[-1]
  _ct12_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct12_arg_m_in
  )
  _ct12_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct12_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct12_arg_r)
  )
  _ct12_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct12_arg_moduli, (int, np.integer)):
    _ct12_arg_moduli = [int(_ct12_arg_moduli)]
  ct12_arg = Polynomial(
      {
          "batch": _ct12_arg_data.shape[0],
          "num_elements": _ct12_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct12_arg_m,
          "precision": 32,
          "degree_layout": (_ct12_arg_r, _ct12_arg_c),
      },
      {"moduli": list(_ct12_arg_moduli)[:_ct12_arg_m]},
  )
  ct12_arg.polynomial = _ct12_arg_data.reshape(
      _ct12_arg_data.shape[0],
      _ct12_arg_data.shape[1],
      _ct12_arg_r,
      _ct12_arg_c,
      _ct12_arg_m_in,
  )[..., :_ct12_arg_m].copy()
  ct12_arg.batch = ct12_arg.polynomial.shape[0]
  ct12_arg.num_elements = ct12_arg.polynomial.shape[1]
  ct12_arg.num_moduli = _ct12_arg_m
  ct12_arg.degree_layout = (_ct12_arg_r, _ct12_arg_c)
  ct12_arg.r = _ct12_arg_r
  ct12_arg.c = _ct12_arg_c
  ct12_arg.moduli = list(_ct12_arg_moduli)[:_ct12_arg_m]
  ct12_arg.moduli_array = jnp.array(
      ct12_arg.moduli, dtype=getattr(ct12_arg, "modulus_dtype", jnp.uint32)
  )
  ct12_pt_ntt = (
      pt6.polynomial[0, 0, :, : ct12_arg.polynomial.shape[-1]]
      .reshape(ct12_arg.r, ct12_arg.c, ct12_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct12_ptct = v0.ptct_mul[v0.max_level]
  ct12_ptct.set_plaintext(ct12_pt_ntt)
  ct12_raw = ct12_ptct.mul(ct12_arg, use_bat=False)
  _ct12_data = (
      ct12_raw.polynomial if hasattr(ct12_raw, "polynomial") else ct12_raw
  )
  _ct12_m_in = _ct12_data.shape[-1]
  _ct12_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct12_m_in
  )
  _ct12_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct12_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct12_r)
  )
  _ct12_moduli = getattr(ct12_raw, "moduli", v0.q_towers)
  if isinstance(_ct12_moduli, (int, np.integer)):
    _ct12_moduli = [int(_ct12_moduli)]
  ct12 = Polynomial(
      {
          "batch": _ct12_data.shape[0],
          "num_elements": _ct12_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct12_m,
          "precision": 32,
          "degree_layout": (_ct12_r, _ct12_c),
      },
      {"moduli": list(_ct12_moduli)[:_ct12_m]},
  )
  ct12.polynomial = _ct12_data.reshape(
      _ct12_data.shape[0], _ct12_data.shape[1], _ct12_r, _ct12_c, _ct12_m_in
  )[..., :_ct12_m].copy()
  ct12.batch = ct12.polynomial.shape[0]
  ct12.num_elements = ct12.polynomial.shape[1]
  ct12.num_moduli = _ct12_m
  ct12.degree_layout = (_ct12_r, _ct12_c)
  ct12.r = _ct12_r
  ct12.c = _ct12_c
  ct12.moduli = list(_ct12_moduli)[:_ct12_m]
  ct12.moduli_array = jnp.array(
      ct12.moduli, dtype=getattr(ct12, "modulus_dtype", jnp.uint32)
  )
  _ct13_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct13_arg_m_in = _ct13_arg_data.shape[-1]
  _ct13_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct13_arg_m_in
  )
  _ct13_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct13_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct13_arg_r)
  )
  _ct13_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct13_arg_moduli, (int, np.integer)):
    _ct13_arg_moduli = [int(_ct13_arg_moduli)]
  ct13_arg = Polynomial(
      {
          "batch": _ct13_arg_data.shape[0],
          "num_elements": _ct13_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct13_arg_m,
          "precision": 32,
          "degree_layout": (_ct13_arg_r, _ct13_arg_c),
      },
      {"moduli": list(_ct13_arg_moduli)[:_ct13_arg_m]},
  )
  ct13_arg.polynomial = _ct13_arg_data.reshape(
      _ct13_arg_data.shape[0],
      _ct13_arg_data.shape[1],
      _ct13_arg_r,
      _ct13_arg_c,
      _ct13_arg_m_in,
  )[..., :_ct13_arg_m].copy()
  ct13_arg.batch = ct13_arg.polynomial.shape[0]
  ct13_arg.num_elements = ct13_arg.polynomial.shape[1]
  ct13_arg.num_moduli = _ct13_arg_m
  ct13_arg.degree_layout = (_ct13_arg_r, _ct13_arg_c)
  ct13_arg.r = _ct13_arg_r
  ct13_arg.c = _ct13_arg_c
  ct13_arg.moduli = list(_ct13_arg_moduli)[:_ct13_arg_m]
  ct13_arg.moduli_array = jnp.array(
      ct13_arg.moduli, dtype=getattr(ct13_arg, "modulus_dtype", jnp.uint32)
  )
  ct13_pt_ntt = (
      pt7.polynomial[0, 0, :, : ct13_arg.polynomial.shape[-1]]
      .reshape(ct13_arg.r, ct13_arg.c, ct13_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct13_ptct = v0.ptct_mul[v0.max_level]
  ct13_ptct.set_plaintext(ct13_pt_ntt)
  ct13_raw = ct13_ptct.mul(ct13_arg, use_bat=False)
  _ct13_data = (
      ct13_raw.polynomial if hasattr(ct13_raw, "polynomial") else ct13_raw
  )
  _ct13_m_in = _ct13_data.shape[-1]
  _ct13_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct13_m_in
  )
  _ct13_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct13_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct13_r)
  )
  _ct13_moduli = getattr(ct13_raw, "moduli", v0.q_towers)
  if isinstance(_ct13_moduli, (int, np.integer)):
    _ct13_moduli = [int(_ct13_moduli)]
  ct13 = Polynomial(
      {
          "batch": _ct13_data.shape[0],
          "num_elements": _ct13_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct13_m,
          "precision": 32,
          "degree_layout": (_ct13_r, _ct13_c),
      },
      {"moduli": list(_ct13_moduli)[:_ct13_m]},
  )
  ct13.polynomial = _ct13_data.reshape(
      _ct13_data.shape[0], _ct13_data.shape[1], _ct13_r, _ct13_c, _ct13_m_in
  )[..., :_ct13_m].copy()
  ct13.batch = ct13.polynomial.shape[0]
  ct13.num_elements = ct13.polynomial.shape[1]
  ct13.num_moduli = _ct13_m
  ct13.degree_layout = (_ct13_r, _ct13_c)
  ct13.r = _ct13_r
  ct13.c = _ct13_c
  ct13.moduli = list(_ct13_moduli)[:_ct13_m]
  ct13.moduli_array = jnp.array(
      ct13.moduli, dtype=getattr(ct13, "modulus_dtype", jnp.uint32)
  )
  _ct14_data = ct12.polynomial if hasattr(ct12, "polynomial") else ct12
  _ct14_m_in = _ct14_data.shape[-1]
  _ct14_m = _ct14_m_in
  _ct14_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct14_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct14_r)
  )
  _ct14_moduli = getattr(ct12, "moduli", v0.q_towers)
  if isinstance(_ct14_moduli, (int, np.integer)):
    _ct14_moduli = [int(_ct14_moduli)]
  ct14 = Polynomial(
      {
          "batch": _ct14_data.shape[0],
          "num_elements": _ct14_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct14_m,
          "precision": 32,
          "degree_layout": (_ct14_r, _ct14_c),
      },
      {"moduli": list(_ct14_moduli)[:_ct14_m]},
  )
  ct14.polynomial = _ct14_data.reshape(
      _ct14_data.shape[0], _ct14_data.shape[1], _ct14_r, _ct14_c, _ct14_m_in
  )[..., :_ct14_m].copy()
  ct14.batch = ct14.polynomial.shape[0]
  ct14.num_elements = ct14.polynomial.shape[1]
  ct14.num_moduli = _ct14_m
  ct14.degree_layout = (_ct14_r, _ct14_c)
  ct14.r = _ct14_r
  ct14.c = _ct14_c
  ct14.moduli = list(_ct14_moduli)[:_ct14_m]
  ct14.moduli_array = jnp.array(
      ct14.moduli, dtype=getattr(ct14, "modulus_dtype", jnp.uint32)
  )
  _ct14_rhs_data = ct13.polynomial if hasattr(ct13, "polynomial") else ct13
  _ct14_rhs_m_in = _ct14_rhs_data.shape[-1]
  _ct14_rhs_m = _ct14_rhs_m_in
  _ct14_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct14_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct14_rhs_r)
  )
  _ct14_rhs_moduli = getattr(ct13, "moduli", v0.q_towers)
  if isinstance(_ct14_rhs_moduli, (int, np.integer)):
    _ct14_rhs_moduli = [int(_ct14_rhs_moduli)]
  ct14_rhs = Polynomial(
      {
          "batch": _ct14_rhs_data.shape[0],
          "num_elements": _ct14_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct14_rhs_m,
          "precision": 32,
          "degree_layout": (_ct14_rhs_r, _ct14_rhs_c),
      },
      {"moduli": list(_ct14_rhs_moduli)[:_ct14_rhs_m]},
  )
  ct14_rhs.polynomial = _ct14_rhs_data.reshape(
      _ct14_rhs_data.shape[0],
      _ct14_rhs_data.shape[1],
      _ct14_rhs_r,
      _ct14_rhs_c,
      _ct14_rhs_m_in,
  )[..., :_ct14_rhs_m].copy()
  ct14_rhs.batch = ct14_rhs.polynomial.shape[0]
  ct14_rhs.num_elements = ct14_rhs.polynomial.shape[1]
  ct14_rhs.num_moduli = _ct14_rhs_m
  ct14_rhs.degree_layout = (_ct14_rhs_r, _ct14_rhs_c)
  ct14_rhs.r = _ct14_rhs_r
  ct14_rhs.c = _ct14_rhs_c
  ct14_rhs.moduli = list(_ct14_rhs_moduli)[:_ct14_rhs_m]
  ct14_rhs.moduli_array = jnp.array(
      ct14_rhs.moduli, dtype=getattr(ct14_rhs, "modulus_dtype", jnp.uint32)
  )
  ct14.add(ct14_rhs)
  _moduli = jnp.array(ct14.moduli, dtype=jnp.uint32)
  ct14.polynomial = jnp.where(
      ct14.polynomial >= _moduli, ct14.polynomial - _moduli, ct14.polynomial
  )
  _ct15_arg_data = ct14.polynomial if hasattr(ct14, "polynomial") else ct14
  _ct15_arg_m_in = _ct15_arg_data.shape[-1]
  _ct15_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct15_arg_m_in
  )
  _ct15_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct15_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct15_arg_r)
  )
  _ct15_arg_moduli = getattr(ct14, "moduli", v0.q_towers)
  if isinstance(_ct15_arg_moduli, (int, np.integer)):
    _ct15_arg_moduli = [int(_ct15_arg_moduli)]
  ct15_arg = Polynomial(
      {
          "batch": _ct15_arg_data.shape[0],
          "num_elements": _ct15_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct15_arg_m,
          "precision": 32,
          "degree_layout": (_ct15_arg_r, _ct15_arg_c),
      },
      {"moduli": list(_ct15_arg_moduli)[:_ct15_arg_m]},
  )
  ct15_arg.polynomial = _ct15_arg_data.reshape(
      _ct15_arg_data.shape[0],
      _ct15_arg_data.shape[1],
      _ct15_arg_r,
      _ct15_arg_c,
      _ct15_arg_m_in,
  )[..., :_ct15_arg_m].copy()
  ct15_arg.batch = ct15_arg.polynomial.shape[0]
  ct15_arg.num_elements = ct15_arg.polynomial.shape[1]
  ct15_arg.num_moduli = _ct15_arg_m
  ct15_arg.degree_layout = (_ct15_arg_r, _ct15_arg_c)
  ct15_arg.r = _ct15_arg_r
  ct15_arg.c = _ct15_arg_c
  ct15_arg.moduli = list(_ct15_arg_moduli)[:_ct15_arg_m]
  ct15_arg.moduli_array = jnp.array(
      ct15_arg.moduli, dtype=getattr(ct15_arg, "modulus_dtype", jnp.uint32)
  )
  ct15_raw = v0.he_rot[v0.max_level, 6].rotate(ct15_arg)
  _ct15_data = (
      ct15_raw.polynomial if hasattr(ct15_raw, "polynomial") else ct15_raw
  )
  _ct15_m_in = _ct15_data.shape[-1]
  _ct15_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct15_m_in
  )
  _ct15_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct15_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct15_r)
  )
  _ct15_moduli = getattr(ct15_raw, "moduli", v0.q_towers)
  if isinstance(_ct15_moduli, (int, np.integer)):
    _ct15_moduli = [int(_ct15_moduli)]
  ct15 = Polynomial(
      {
          "batch": _ct15_data.shape[0],
          "num_elements": _ct15_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct15_m,
          "precision": 32,
          "degree_layout": (_ct15_r, _ct15_c),
      },
      {"moduli": list(_ct15_moduli)[:_ct15_m]},
  )
  ct15.polynomial = _ct15_data.reshape(
      _ct15_data.shape[0], _ct15_data.shape[1], _ct15_r, _ct15_c, _ct15_m_in
  )[..., :_ct15_m].copy()
  ct15.batch = ct15.polynomial.shape[0]
  ct15.num_elements = ct15.polynomial.shape[1]
  ct15.num_moduli = _ct15_m
  ct15.degree_layout = (_ct15_r, _ct15_c)
  ct15.r = _ct15_r
  ct15.c = _ct15_c
  ct15.moduli = list(_ct15_moduli)[:_ct15_m]
  ct15.moduli_array = jnp.array(
      ct15.moduli, dtype=getattr(ct15, "modulus_dtype", jnp.uint32)
  )
  _ct16_data = ct1.polynomial if hasattr(ct1, "polynomial") else ct1
  _ct16_m_in = _ct16_data.shape[-1]
  _ct16_m = _ct16_m_in
  _ct16_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct16_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct16_r)
  )
  _ct16_moduli = getattr(ct1, "moduli", v0.q_towers)
  if isinstance(_ct16_moduli, (int, np.integer)):
    _ct16_moduli = [int(_ct16_moduli)]
  ct16 = Polynomial(
      {
          "batch": _ct16_data.shape[0],
          "num_elements": _ct16_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct16_m,
          "precision": 32,
          "degree_layout": (_ct16_r, _ct16_c),
      },
      {"moduli": list(_ct16_moduli)[:_ct16_m]},
  )
  ct16.polynomial = _ct16_data.reshape(
      _ct16_data.shape[0], _ct16_data.shape[1], _ct16_r, _ct16_c, _ct16_m_in
  )[..., :_ct16_m].copy()
  ct16.batch = ct16.polynomial.shape[0]
  ct16.num_elements = ct16.polynomial.shape[1]
  ct16.num_moduli = _ct16_m
  ct16.degree_layout = (_ct16_r, _ct16_c)
  ct16.r = _ct16_r
  ct16.c = _ct16_c
  ct16.moduli = list(_ct16_moduli)[:_ct16_m]
  ct16.moduli_array = jnp.array(
      ct16.moduli, dtype=getattr(ct16, "modulus_dtype", jnp.uint32)
  )
  _ct16_rhs_data = ct3.polynomial if hasattr(ct3, "polynomial") else ct3
  _ct16_rhs_m_in = _ct16_rhs_data.shape[-1]
  _ct16_rhs_m = _ct16_rhs_m_in
  _ct16_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct16_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct16_rhs_r)
  )
  _ct16_rhs_moduli = getattr(ct3, "moduli", v0.q_towers)
  if isinstance(_ct16_rhs_moduli, (int, np.integer)):
    _ct16_rhs_moduli = [int(_ct16_rhs_moduli)]
  ct16_rhs = Polynomial(
      {
          "batch": _ct16_rhs_data.shape[0],
          "num_elements": _ct16_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct16_rhs_m,
          "precision": 32,
          "degree_layout": (_ct16_rhs_r, _ct16_rhs_c),
      },
      {"moduli": list(_ct16_rhs_moduli)[:_ct16_rhs_m]},
  )
  ct16_rhs.polynomial = _ct16_rhs_data.reshape(
      _ct16_rhs_data.shape[0],
      _ct16_rhs_data.shape[1],
      _ct16_rhs_r,
      _ct16_rhs_c,
      _ct16_rhs_m_in,
  )[..., :_ct16_rhs_m].copy()
  ct16_rhs.batch = ct16_rhs.polynomial.shape[0]
  ct16_rhs.num_elements = ct16_rhs.polynomial.shape[1]
  ct16_rhs.num_moduli = _ct16_rhs_m
  ct16_rhs.degree_layout = (_ct16_rhs_r, _ct16_rhs_c)
  ct16_rhs.r = _ct16_rhs_r
  ct16_rhs.c = _ct16_rhs_c
  ct16_rhs.moduli = list(_ct16_rhs_moduli)[:_ct16_rhs_m]
  ct16_rhs.moduli_array = jnp.array(
      ct16_rhs.moduli, dtype=getattr(ct16_rhs, "modulus_dtype", jnp.uint32)
  )
  ct16.add(ct16_rhs)
  _moduli = jnp.array(ct16.moduli, dtype=jnp.uint32)
  ct16.polynomial = jnp.where(
      ct16.polynomial >= _moduli, ct16.polynomial - _moduli, ct16.polynomial
  )
  _ct17_data = ct5.polynomial if hasattr(ct5, "polynomial") else ct5
  _ct17_m_in = _ct17_data.shape[-1]
  _ct17_m = _ct17_m_in
  _ct17_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct17_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct17_r)
  )
  _ct17_moduli = getattr(ct5, "moduli", v0.q_towers)
  if isinstance(_ct17_moduli, (int, np.integer)):
    _ct17_moduli = [int(_ct17_moduli)]
  ct17 = Polynomial(
      {
          "batch": _ct17_data.shape[0],
          "num_elements": _ct17_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct17_m,
          "precision": 32,
          "degree_layout": (_ct17_r, _ct17_c),
      },
      {"moduli": list(_ct17_moduli)[:_ct17_m]},
  )
  ct17.polynomial = _ct17_data.reshape(
      _ct17_data.shape[0], _ct17_data.shape[1], _ct17_r, _ct17_c, _ct17_m_in
  )[..., :_ct17_m].copy()
  ct17.batch = ct17.polynomial.shape[0]
  ct17.num_elements = ct17.polynomial.shape[1]
  ct17.num_moduli = _ct17_m
  ct17.degree_layout = (_ct17_r, _ct17_c)
  ct17.r = _ct17_r
  ct17.c = _ct17_c
  ct17.moduli = list(_ct17_moduli)[:_ct17_m]
  ct17.moduli_array = jnp.array(
      ct17.moduli, dtype=getattr(ct17, "modulus_dtype", jnp.uint32)
  )
  _ct17_rhs_data = ct11.polynomial if hasattr(ct11, "polynomial") else ct11
  _ct17_rhs_m_in = _ct17_rhs_data.shape[-1]
  _ct17_rhs_m = _ct17_rhs_m_in
  _ct17_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct17_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct17_rhs_r)
  )
  _ct17_rhs_moduli = getattr(ct11, "moduli", v0.q_towers)
  if isinstance(_ct17_rhs_moduli, (int, np.integer)):
    _ct17_rhs_moduli = [int(_ct17_rhs_moduli)]
  ct17_rhs = Polynomial(
      {
          "batch": _ct17_rhs_data.shape[0],
          "num_elements": _ct17_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct17_rhs_m,
          "precision": 32,
          "degree_layout": (_ct17_rhs_r, _ct17_rhs_c),
      },
      {"moduli": list(_ct17_rhs_moduli)[:_ct17_rhs_m]},
  )
  ct17_rhs.polynomial = _ct17_rhs_data.reshape(
      _ct17_rhs_data.shape[0],
      _ct17_rhs_data.shape[1],
      _ct17_rhs_r,
      _ct17_rhs_c,
      _ct17_rhs_m_in,
  )[..., :_ct17_rhs_m].copy()
  ct17_rhs.batch = ct17_rhs.polynomial.shape[0]
  ct17_rhs.num_elements = ct17_rhs.polynomial.shape[1]
  ct17_rhs.num_moduli = _ct17_rhs_m
  ct17_rhs.degree_layout = (_ct17_rhs_r, _ct17_rhs_c)
  ct17_rhs.r = _ct17_rhs_r
  ct17_rhs.c = _ct17_rhs_c
  ct17_rhs.moduli = list(_ct17_rhs_moduli)[:_ct17_rhs_m]
  ct17_rhs.moduli_array = jnp.array(
      ct17_rhs.moduli, dtype=getattr(ct17_rhs, "modulus_dtype", jnp.uint32)
  )
  ct17.add(ct17_rhs)
  _moduli = jnp.array(ct17.moduli, dtype=jnp.uint32)
  ct17.polynomial = jnp.where(
      ct17.polynomial >= _moduli, ct17.polynomial - _moduli, ct17.polynomial
  )
  _ct18_data = ct17.polynomial if hasattr(ct17, "polynomial") else ct17
  _ct18_m_in = _ct18_data.shape[-1]
  _ct18_m = _ct18_m_in
  _ct18_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct18_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct18_r)
  )
  _ct18_moduli = getattr(ct17, "moduli", v0.q_towers)
  if isinstance(_ct18_moduli, (int, np.integer)):
    _ct18_moduli = [int(_ct18_moduli)]
  ct18 = Polynomial(
      {
          "batch": _ct18_data.shape[0],
          "num_elements": _ct18_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct18_m,
          "precision": 32,
          "degree_layout": (_ct18_r, _ct18_c),
      },
      {"moduli": list(_ct18_moduli)[:_ct18_m]},
  )
  ct18.polynomial = _ct18_data.reshape(
      _ct18_data.shape[0], _ct18_data.shape[1], _ct18_r, _ct18_c, _ct18_m_in
  )[..., :_ct18_m].copy()
  ct18.batch = ct18.polynomial.shape[0]
  ct18.num_elements = ct18.polynomial.shape[1]
  ct18.num_moduli = _ct18_m
  ct18.degree_layout = (_ct18_r, _ct18_c)
  ct18.r = _ct18_r
  ct18.c = _ct18_c
  ct18.moduli = list(_ct18_moduli)[:_ct18_m]
  ct18.moduli_array = jnp.array(
      ct18.moduli, dtype=getattr(ct18, "modulus_dtype", jnp.uint32)
  )
  _ct18_rhs_data = ct15.polynomial if hasattr(ct15, "polynomial") else ct15
  _ct18_rhs_m_in = _ct18_rhs_data.shape[-1]
  _ct18_rhs_m = _ct18_rhs_m_in
  _ct18_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct18_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct18_rhs_r)
  )
  _ct18_rhs_moduli = getattr(ct15, "moduli", v0.q_towers)
  if isinstance(_ct18_rhs_moduli, (int, np.integer)):
    _ct18_rhs_moduli = [int(_ct18_rhs_moduli)]
  ct18_rhs = Polynomial(
      {
          "batch": _ct18_rhs_data.shape[0],
          "num_elements": _ct18_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct18_rhs_m,
          "precision": 32,
          "degree_layout": (_ct18_rhs_r, _ct18_rhs_c),
      },
      {"moduli": list(_ct18_rhs_moduli)[:_ct18_rhs_m]},
  )
  ct18_rhs.polynomial = _ct18_rhs_data.reshape(
      _ct18_rhs_data.shape[0],
      _ct18_rhs_data.shape[1],
      _ct18_rhs_r,
      _ct18_rhs_c,
      _ct18_rhs_m_in,
  )[..., :_ct18_rhs_m].copy()
  ct18_rhs.batch = ct18_rhs.polynomial.shape[0]
  ct18_rhs.num_elements = ct18_rhs.polynomial.shape[1]
  ct18_rhs.num_moduli = _ct18_rhs_m
  ct18_rhs.degree_layout = (_ct18_rhs_r, _ct18_rhs_c)
  ct18_rhs.r = _ct18_rhs_r
  ct18_rhs.c = _ct18_rhs_c
  ct18_rhs.moduli = list(_ct18_rhs_moduli)[:_ct18_rhs_m]
  ct18_rhs.moduli_array = jnp.array(
      ct18_rhs.moduli, dtype=getattr(ct18_rhs, "modulus_dtype", jnp.uint32)
  )
  ct18.add(ct18_rhs)
  _moduli = jnp.array(ct18.moduli, dtype=jnp.uint32)
  ct18.polynomial = jnp.where(
      ct18.polynomial >= _moduli, ct18.polynomial - _moduli, ct18.polynomial
  )
  _ct19_data = ct16.polynomial if hasattr(ct16, "polynomial") else ct16
  _ct19_m_in = _ct19_data.shape[-1]
  _ct19_m = _ct19_m_in
  _ct19_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct19_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct19_r)
  )
  _ct19_moduli = getattr(ct16, "moduli", v0.q_towers)
  if isinstance(_ct19_moduli, (int, np.integer)):
    _ct19_moduli = [int(_ct19_moduli)]
  ct19 = Polynomial(
      {
          "batch": _ct19_data.shape[0],
          "num_elements": _ct19_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct19_m,
          "precision": 32,
          "degree_layout": (_ct19_r, _ct19_c),
      },
      {"moduli": list(_ct19_moduli)[:_ct19_m]},
  )
  ct19.polynomial = _ct19_data.reshape(
      _ct19_data.shape[0], _ct19_data.shape[1], _ct19_r, _ct19_c, _ct19_m_in
  )[..., :_ct19_m].copy()
  ct19.batch = ct19.polynomial.shape[0]
  ct19.num_elements = ct19.polynomial.shape[1]
  ct19.num_moduli = _ct19_m
  ct19.degree_layout = (_ct19_r, _ct19_c)
  ct19.r = _ct19_r
  ct19.c = _ct19_c
  ct19.moduli = list(_ct19_moduli)[:_ct19_m]
  ct19.moduli_array = jnp.array(
      ct19.moduli, dtype=getattr(ct19, "modulus_dtype", jnp.uint32)
  )
  _ct19_rhs_data = ct18.polynomial if hasattr(ct18, "polynomial") else ct18
  _ct19_rhs_m_in = _ct19_rhs_data.shape[-1]
  _ct19_rhs_m = _ct19_rhs_m_in
  _ct19_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct19_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct19_rhs_r)
  )
  _ct19_rhs_moduli = getattr(ct18, "moduli", v0.q_towers)
  if isinstance(_ct19_rhs_moduli, (int, np.integer)):
    _ct19_rhs_moduli = [int(_ct19_rhs_moduli)]
  ct19_rhs = Polynomial(
      {
          "batch": _ct19_rhs_data.shape[0],
          "num_elements": _ct19_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct19_rhs_m,
          "precision": 32,
          "degree_layout": (_ct19_rhs_r, _ct19_rhs_c),
      },
      {"moduli": list(_ct19_rhs_moduli)[:_ct19_rhs_m]},
  )
  ct19_rhs.polynomial = _ct19_rhs_data.reshape(
      _ct19_rhs_data.shape[0],
      _ct19_rhs_data.shape[1],
      _ct19_rhs_r,
      _ct19_rhs_c,
      _ct19_rhs_m_in,
  )[..., :_ct19_rhs_m].copy()
  ct19_rhs.batch = ct19_rhs.polynomial.shape[0]
  ct19_rhs.num_elements = ct19_rhs.polynomial.shape[1]
  ct19_rhs.num_moduli = _ct19_rhs_m
  ct19_rhs.degree_layout = (_ct19_rhs_r, _ct19_rhs_c)
  ct19_rhs.r = _ct19_rhs_r
  ct19_rhs.c = _ct19_rhs_c
  ct19_rhs.moduli = list(_ct19_rhs_moduli)[:_ct19_rhs_m]
  ct19_rhs.moduli_array = jnp.array(
      ct19_rhs.moduli, dtype=getattr(ct19_rhs, "modulus_dtype", jnp.uint32)
  )
  ct19.add(ct19_rhs)
  _moduli = jnp.array(ct19.moduli, dtype=jnp.uint32)
  ct19.polynomial = jnp.where(
      ct19.polynomial >= _moduli, ct19.polynomial - _moduli, ct19.polynomial
  )
  v16 = [None] * 1
  _ct20_arg_data = ct19.polynomial if hasattr(ct19, "polynomial") else ct19
  _ct20_arg_m_in = _ct20_arg_data.shape[-1]
  _ct20_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct20_arg_m_in
  )
  _ct20_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct20_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct20_arg_r)
  )
  _ct20_arg_moduli = getattr(ct19, "moduli", v0.q_towers)
  if isinstance(_ct20_arg_moduli, (int, np.integer)):
    _ct20_arg_moduli = [int(_ct20_arg_moduli)]
  ct20_arg = Polynomial(
      {
          "batch": _ct20_arg_data.shape[0],
          "num_elements": _ct20_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct20_arg_m,
          "precision": 32,
          "degree_layout": (_ct20_arg_r, _ct20_arg_c),
      },
      {"moduli": list(_ct20_arg_moduli)[:_ct20_arg_m]},
  )
  ct20_arg.polynomial = _ct20_arg_data.reshape(
      _ct20_arg_data.shape[0],
      _ct20_arg_data.shape[1],
      _ct20_arg_r,
      _ct20_arg_c,
      _ct20_arg_m_in,
  )[..., :_ct20_arg_m].copy()
  ct20_arg.batch = ct20_arg.polynomial.shape[0]
  ct20_arg.num_elements = ct20_arg.polynomial.shape[1]
  ct20_arg.num_moduli = _ct20_arg_m
  ct20_arg.degree_layout = (_ct20_arg_r, _ct20_arg_c)
  ct20_arg.r = _ct20_arg_r
  ct20_arg.c = _ct20_arg_c
  ct20_arg.moduli = list(_ct20_arg_moduli)[:_ct20_arg_m]
  ct20_arg.moduli_array = jnp.array(
      ct20_arg.moduli, dtype=getattr(ct20_arg, "modulus_dtype", jnp.uint32)
  )
  ct20_raw = v0.he_rescale[v0.max_level, v0.max_level - 1](ct20_arg)
  _ct20_data = (
      ct20_raw.polynomial if hasattr(ct20_raw, "polynomial") else ct20_raw
  )
  _ct20_m_in = _ct20_data.shape[-1]
  _ct20_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct20_m_in
  )
  _ct20_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct20_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct20_r)
  )
  _ct20_moduli = getattr(ct20_raw, "moduli", v0.q_towers)
  if isinstance(_ct20_moduli, (int, np.integer)):
    _ct20_moduli = [int(_ct20_moduli)]
  ct20 = Polynomial(
      {
          "batch": _ct20_data.shape[0],
          "num_elements": _ct20_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct20_m,
          "precision": 32,
          "degree_layout": (_ct20_r, _ct20_c),
      },
      {"moduli": list(_ct20_moduli)[:_ct20_m]},
  )
  ct20.polynomial = _ct20_data.reshape(
      _ct20_data.shape[0], _ct20_data.shape[1], _ct20_r, _ct20_c, _ct20_m_in
  )[..., :_ct20_m].copy()
  ct20.batch = ct20.polynomial.shape[0]
  ct20.num_elements = ct20.polynomial.shape[1]
  ct20.num_moduli = _ct20_m
  ct20.degree_layout = (_ct20_r, _ct20_c)
  ct20.r = _ct20_r
  ct20.c = _ct20_c
  ct20.moduli = list(_ct20_moduli)[:_ct20_m]
  ct20.moduli_array = jnp.array(
      ct20.moduli, dtype=getattr(ct20, "modulus_dtype", jnp.uint32)
  )
  v16[0] = ct20
  v17 = v16
  return v17


def matvec_identity(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  (v3, v4, v5, v6, v7, v8, v9, v10) = matvec_identity__preprocessing(v0, v1)
  v11 = matvec_identity__preprocessed(
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10
  )
  return v11


def matvec_identity__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v15)
  v0.public_key = v3
  ct_raw = v0.encrypt(pt)
  _ct_data = ct_raw.polynomial if hasattr(ct_raw, "polynomial") else ct_raw
  _ct_m_in = _ct_data.shape[-1]
  _ct_m = _ct_m_in
  _ct_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct_r)
  )
  _ct_moduli = getattr(ct_raw, "moduli", v0.q_towers)
  if isinstance(_ct_moduli, (int, np.integer)):
    _ct_moduli = [int(_ct_moduli)]
  ct = Polynomial(
      {
          "batch": _ct_data.shape[0],
          "num_elements": _ct_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct_m,
          "precision": 32,
          "degree_layout": (_ct_r, _ct_c),
      },
      {"moduli": list(_ct_moduli)[:_ct_m]},
  )
  ct.polynomial = _ct_data.reshape(
      _ct_data.shape[0], _ct_data.shape[1], _ct_r, _ct_c, _ct_m_in
  )[..., :_ct_m].copy()
  ct.batch = ct.polynomial.shape[0]
  ct.num_elements = ct.polynomial.shape[1]
  ct.num_moduli = _ct_m
  ct.degree_layout = (_ct_r, _ct_c)
  ct.r = _ct_r
  ct.c = _ct_c
  ct.moduli = list(_ct_moduli)[:_ct_m]
  ct.moduli_array = jnp.array(
      ct.moduli, dtype=getattr(ct, "modulus_dtype", jnp.uint32)
  )
  v16 = [ct]
  return v16


def matvec_identity__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 0
  v8 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  _num_moduli = ct.polynomial.shape[-1]
  _q_sub = list(getattr(ct, "moduli", v0.q_towers))[:_num_moduli]
  _ct_for_dec = Polynomial(
      {
          "batch": ct.polynomial.shape[0],
          "num_elements": ct.polynomial.shape[1],
          "degree": v0.degree,
          "precision": 32,
          "num_moduli": _num_moduli,
          "degree_layout": (v0.degree,),
      },
      {"moduli": _q_sub},
  )
  _ct_for_dec.set_batch_polynomial(
      ct.polynomial.reshape(
          ct.polynomial.shape[0], ct.polynomial.shape[1], v0.degree, _num_moduli
      )
  )
  pt = v0.decrypt(_ct_for_dec)
  v9 = v0.decode(pt, is_ntt=False).real.reshape(1, 8)
  v10 = v8.copy()
  for v11 in range(0, 8):
    v13 = int(v11)
    v14 = v9[0, v13]
    v10[v13] = v14
  return v10


def matvec_shift__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> (
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
):
  v2 = np.array(
      [
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          1.000000e00,
          1.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
          0.000000e00,
      ],
      dtype=np.float32,
  ).reshape(8, 8)
  v3 = _assign_layout_15335824159471298539(v2)
  v4 = v3[3 : 3 + 1, 0 : 0 + 5]
  v5 = v3[3 : 3 + 1, 5 : 5 + 3]
  v6 = np.zeros(
      (
          1,
          8,
      ),
      dtype=np.float32,
  )
  v7 = v6.copy()
  v7[0 : 0 + 1, 3 : 3 + 5] = v4
  v8 = v7.copy()
  v8[0 : 0 + 1, 0 : 0 + 3] = v5
  v9 = v3[4 : 4 + 1, 0 : 0 + 5]
  v10 = v3[4 : 4 + 1, 5 : 5 + 3]
  v11 = v6.copy()
  v11[0 : 0 + 1, 3 : 3 + 5] = v9
  v12 = v11.copy()
  v12[0 : 0 + 1, 0 : 0 + 3] = v10
  v13 = v3[5 : 5 + 1, 0 : 0 + 5]
  v14 = v3[5 : 5 + 1, 5 : 5 + 3]
  v15 = v6.copy()
  v15[0 : 0 + 1, 3 : 3 + 5] = v13
  v16 = v15.copy()
  v16[0 : 0 + 1, 0 : 0 + 3] = v14
  v17 = v3[6 : 6 + 1, 0 : 0 + 2]
  v18 = v3[6 : 6 + 1, 2 : 2 + 6]
  v19 = v6.copy()
  v19[0 : 0 + 1, 6 : 6 + 2] = v17
  v20 = v19.copy()
  v20[0 : 0 + 1, 0 : 0 + 6] = v18
  v21 = v3[7 : 7 + 1, 0 : 0 + 2]
  v22 = v3[7 : 7 + 1, 2 : 2 + 6]
  v23 = v6.copy()
  v23[0 : 0 + 1, 6 : 6 + 2] = v21
  v24 = v23.copy()
  v24[0 : 0 + 1, 0 : 0 + 6] = v22
  v25 = v3[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v25)
  v26 = v3[1 : 1 + 1, 0 : 0 + 8].reshape(8)
  pt1 = v0.encode(v26)
  v27 = v3[2 : 2 + 1, 0 : 0 + 8].reshape(8)
  pt2 = v0.encode(v27)
  v28 = v8[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt3 = v0.encode(v28)
  v29 = v12[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt4 = v0.encode(v29)
  v30 = v16[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt5 = v0.encode(v30)
  v31 = v20[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt6 = v0.encode(v31)
  v32 = v24[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt7 = v0.encode(v32)
  v33 = [pt]
  v34 = [pt1]
  v35 = [pt2]
  v36 = [pt3]
  v37 = [pt4]
  v38 = [pt5]
  v39 = [pt6]
  v40 = [pt7]
  return (v33, v34, v35, v36, v37, v38, v39, v40)


def matvec_shift__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
    v4: np.ndarray,
    v5: np.ndarray,
    v6: np.ndarray,
    v7: np.ndarray,
    v8: np.ndarray,
    v9: np.ndarray,
    v10: np.ndarray,
) -> np.ndarray:
  v11 = 1
  v12 = 2
  v13 = 3
  v14 = 6
  v15 = 0
  pt = v3[0]
  pt1 = v4[0]
  pt2 = v5[0]
  pt3 = v6[0]
  pt4 = v7[0]
  pt5 = v8[0]
  pt6 = v9[0]
  pt7 = v10[0]
  ct = v2[0]
  _ct1_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct1_arg_m_in = _ct1_arg_data.shape[-1]
  _ct1_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct1_arg_m_in
  )
  _ct1_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct1_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct1_arg_r)
  )
  _ct1_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct1_arg_moduli, (int, np.integer)):
    _ct1_arg_moduli = [int(_ct1_arg_moduli)]
  ct1_arg = Polynomial(
      {
          "batch": _ct1_arg_data.shape[0],
          "num_elements": _ct1_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct1_arg_m,
          "precision": 32,
          "degree_layout": (_ct1_arg_r, _ct1_arg_c),
      },
      {"moduli": list(_ct1_arg_moduli)[:_ct1_arg_m]},
  )
  ct1_arg.polynomial = _ct1_arg_data.reshape(
      _ct1_arg_data.shape[0],
      _ct1_arg_data.shape[1],
      _ct1_arg_r,
      _ct1_arg_c,
      _ct1_arg_m_in,
  )[..., :_ct1_arg_m].copy()
  ct1_arg.batch = ct1_arg.polynomial.shape[0]
  ct1_arg.num_elements = ct1_arg.polynomial.shape[1]
  ct1_arg.num_moduli = _ct1_arg_m
  ct1_arg.degree_layout = (_ct1_arg_r, _ct1_arg_c)
  ct1_arg.r = _ct1_arg_r
  ct1_arg.c = _ct1_arg_c
  ct1_arg.moduli = list(_ct1_arg_moduli)[:_ct1_arg_m]
  ct1_arg.moduli_array = jnp.array(
      ct1_arg.moduli, dtype=getattr(ct1_arg, "modulus_dtype", jnp.uint32)
  )
  ct1_pt_ntt = (
      pt.polynomial[0, 0, :, : ct1_arg.polynomial.shape[-1]]
      .reshape(ct1_arg.r, ct1_arg.c, ct1_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct1_ptct = v0.ptct_mul[v0.max_level]
  ct1_ptct.set_plaintext(ct1_pt_ntt)
  ct1_raw = ct1_ptct.mul(ct1_arg, use_bat=False)
  _ct1_data = ct1_raw.polynomial if hasattr(ct1_raw, "polynomial") else ct1_raw
  _ct1_m_in = _ct1_data.shape[-1]
  _ct1_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct1_m_in
  )
  _ct1_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct1_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct1_r)
  )
  _ct1_moduli = getattr(ct1_raw, "moduli", v0.q_towers)
  if isinstance(_ct1_moduli, (int, np.integer)):
    _ct1_moduli = [int(_ct1_moduli)]
  ct1 = Polynomial(
      {
          "batch": _ct1_data.shape[0],
          "num_elements": _ct1_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct1_m,
          "precision": 32,
          "degree_layout": (_ct1_r, _ct1_c),
      },
      {"moduli": list(_ct1_moduli)[:_ct1_m]},
  )
  ct1.polynomial = _ct1_data.reshape(
      _ct1_data.shape[0], _ct1_data.shape[1], _ct1_r, _ct1_c, _ct1_m_in
  )[..., :_ct1_m].copy()
  ct1.batch = ct1.polynomial.shape[0]
  ct1.num_elements = ct1.polynomial.shape[1]
  ct1.num_moduli = _ct1_m
  ct1.degree_layout = (_ct1_r, _ct1_c)
  ct1.r = _ct1_r
  ct1.c = _ct1_c
  ct1.moduli = list(_ct1_moduli)[:_ct1_m]
  ct1.moduli_array = jnp.array(
      ct1.moduli, dtype=getattr(ct1, "modulus_dtype", jnp.uint32)
  )
  _ct2_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct2_arg_m_in = _ct2_arg_data.shape[-1]
  _ct2_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct2_arg_m_in
  )
  _ct2_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct2_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct2_arg_r)
  )
  _ct2_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct2_arg_moduli, (int, np.integer)):
    _ct2_arg_moduli = [int(_ct2_arg_moduli)]
  ct2_arg = Polynomial(
      {
          "batch": _ct2_arg_data.shape[0],
          "num_elements": _ct2_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct2_arg_m,
          "precision": 32,
          "degree_layout": (_ct2_arg_r, _ct2_arg_c),
      },
      {"moduli": list(_ct2_arg_moduli)[:_ct2_arg_m]},
  )
  ct2_arg.polynomial = _ct2_arg_data.reshape(
      _ct2_arg_data.shape[0],
      _ct2_arg_data.shape[1],
      _ct2_arg_r,
      _ct2_arg_c,
      _ct2_arg_m_in,
  )[..., :_ct2_arg_m].copy()
  ct2_arg.batch = ct2_arg.polynomial.shape[0]
  ct2_arg.num_elements = ct2_arg.polynomial.shape[1]
  ct2_arg.num_moduli = _ct2_arg_m
  ct2_arg.degree_layout = (_ct2_arg_r, _ct2_arg_c)
  ct2_arg.r = _ct2_arg_r
  ct2_arg.c = _ct2_arg_c
  ct2_arg.moduli = list(_ct2_arg_moduli)[:_ct2_arg_m]
  ct2_arg.moduli_array = jnp.array(
      ct2_arg.moduli, dtype=getattr(ct2_arg, "modulus_dtype", jnp.uint32)
  )
  ct2_raw = v0.he_rot[v0.max_level, 1].rotate(ct2_arg)
  _ct2_data = ct2_raw.polynomial if hasattr(ct2_raw, "polynomial") else ct2_raw
  _ct2_m_in = _ct2_data.shape[-1]
  _ct2_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct2_m_in
  )
  _ct2_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct2_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct2_r)
  )
  _ct2_moduli = getattr(ct2_raw, "moduli", v0.q_towers)
  if isinstance(_ct2_moduli, (int, np.integer)):
    _ct2_moduli = [int(_ct2_moduli)]
  ct2 = Polynomial(
      {
          "batch": _ct2_data.shape[0],
          "num_elements": _ct2_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct2_m,
          "precision": 32,
          "degree_layout": (_ct2_r, _ct2_c),
      },
      {"moduli": list(_ct2_moduli)[:_ct2_m]},
  )
  ct2.polynomial = _ct2_data.reshape(
      _ct2_data.shape[0], _ct2_data.shape[1], _ct2_r, _ct2_c, _ct2_m_in
  )[..., :_ct2_m].copy()
  ct2.batch = ct2.polynomial.shape[0]
  ct2.num_elements = ct2.polynomial.shape[1]
  ct2.num_moduli = _ct2_m
  ct2.degree_layout = (_ct2_r, _ct2_c)
  ct2.r = _ct2_r
  ct2.c = _ct2_c
  ct2.moduli = list(_ct2_moduli)[:_ct2_m]
  ct2.moduli_array = jnp.array(
      ct2.moduli, dtype=getattr(ct2, "modulus_dtype", jnp.uint32)
  )
  _ct3_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct3_arg_m_in = _ct3_arg_data.shape[-1]
  _ct3_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct3_arg_m_in
  )
  _ct3_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct3_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct3_arg_r)
  )
  _ct3_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct3_arg_moduli, (int, np.integer)):
    _ct3_arg_moduli = [int(_ct3_arg_moduli)]
  ct3_arg = Polynomial(
      {
          "batch": _ct3_arg_data.shape[0],
          "num_elements": _ct3_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct3_arg_m,
          "precision": 32,
          "degree_layout": (_ct3_arg_r, _ct3_arg_c),
      },
      {"moduli": list(_ct3_arg_moduli)[:_ct3_arg_m]},
  )
  ct3_arg.polynomial = _ct3_arg_data.reshape(
      _ct3_arg_data.shape[0],
      _ct3_arg_data.shape[1],
      _ct3_arg_r,
      _ct3_arg_c,
      _ct3_arg_m_in,
  )[..., :_ct3_arg_m].copy()
  ct3_arg.batch = ct3_arg.polynomial.shape[0]
  ct3_arg.num_elements = ct3_arg.polynomial.shape[1]
  ct3_arg.num_moduli = _ct3_arg_m
  ct3_arg.degree_layout = (_ct3_arg_r, _ct3_arg_c)
  ct3_arg.r = _ct3_arg_r
  ct3_arg.c = _ct3_arg_c
  ct3_arg.moduli = list(_ct3_arg_moduli)[:_ct3_arg_m]
  ct3_arg.moduli_array = jnp.array(
      ct3_arg.moduli, dtype=getattr(ct3_arg, "modulus_dtype", jnp.uint32)
  )
  ct3_pt_ntt = (
      pt1.polynomial[0, 0, :, : ct3_arg.polynomial.shape[-1]]
      .reshape(ct3_arg.r, ct3_arg.c, ct3_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct3_ptct = v0.ptct_mul[v0.max_level]
  ct3_ptct.set_plaintext(ct3_pt_ntt)
  ct3_raw = ct3_ptct.mul(ct3_arg, use_bat=False)
  _ct3_data = ct3_raw.polynomial if hasattr(ct3_raw, "polynomial") else ct3_raw
  _ct3_m_in = _ct3_data.shape[-1]
  _ct3_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct3_m_in
  )
  _ct3_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct3_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct3_r)
  )
  _ct3_moduli = getattr(ct3_raw, "moduli", v0.q_towers)
  if isinstance(_ct3_moduli, (int, np.integer)):
    _ct3_moduli = [int(_ct3_moduli)]
  ct3 = Polynomial(
      {
          "batch": _ct3_data.shape[0],
          "num_elements": _ct3_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct3_m,
          "precision": 32,
          "degree_layout": (_ct3_r, _ct3_c),
      },
      {"moduli": list(_ct3_moduli)[:_ct3_m]},
  )
  ct3.polynomial = _ct3_data.reshape(
      _ct3_data.shape[0], _ct3_data.shape[1], _ct3_r, _ct3_c, _ct3_m_in
  )[..., :_ct3_m].copy()
  ct3.batch = ct3.polynomial.shape[0]
  ct3.num_elements = ct3.polynomial.shape[1]
  ct3.num_moduli = _ct3_m
  ct3.degree_layout = (_ct3_r, _ct3_c)
  ct3.r = _ct3_r
  ct3.c = _ct3_c
  ct3.moduli = list(_ct3_moduli)[:_ct3_m]
  ct3.moduli_array = jnp.array(
      ct3.moduli, dtype=getattr(ct3, "modulus_dtype", jnp.uint32)
  )
  _ct4_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct4_arg_m_in = _ct4_arg_data.shape[-1]
  _ct4_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct4_arg_m_in
  )
  _ct4_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct4_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct4_arg_r)
  )
  _ct4_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct4_arg_moduli, (int, np.integer)):
    _ct4_arg_moduli = [int(_ct4_arg_moduli)]
  ct4_arg = Polynomial(
      {
          "batch": _ct4_arg_data.shape[0],
          "num_elements": _ct4_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct4_arg_m,
          "precision": 32,
          "degree_layout": (_ct4_arg_r, _ct4_arg_c),
      },
      {"moduli": list(_ct4_arg_moduli)[:_ct4_arg_m]},
  )
  ct4_arg.polynomial = _ct4_arg_data.reshape(
      _ct4_arg_data.shape[0],
      _ct4_arg_data.shape[1],
      _ct4_arg_r,
      _ct4_arg_c,
      _ct4_arg_m_in,
  )[..., :_ct4_arg_m].copy()
  ct4_arg.batch = ct4_arg.polynomial.shape[0]
  ct4_arg.num_elements = ct4_arg.polynomial.shape[1]
  ct4_arg.num_moduli = _ct4_arg_m
  ct4_arg.degree_layout = (_ct4_arg_r, _ct4_arg_c)
  ct4_arg.r = _ct4_arg_r
  ct4_arg.c = _ct4_arg_c
  ct4_arg.moduli = list(_ct4_arg_moduli)[:_ct4_arg_m]
  ct4_arg.moduli_array = jnp.array(
      ct4_arg.moduli, dtype=getattr(ct4_arg, "modulus_dtype", jnp.uint32)
  )
  ct4_raw = v0.he_rot[v0.max_level, 2].rotate(ct4_arg)
  _ct4_data = ct4_raw.polynomial if hasattr(ct4_raw, "polynomial") else ct4_raw
  _ct4_m_in = _ct4_data.shape[-1]
  _ct4_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct4_m_in
  )
  _ct4_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct4_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct4_r)
  )
  _ct4_moduli = getattr(ct4_raw, "moduli", v0.q_towers)
  if isinstance(_ct4_moduli, (int, np.integer)):
    _ct4_moduli = [int(_ct4_moduli)]
  ct4 = Polynomial(
      {
          "batch": _ct4_data.shape[0],
          "num_elements": _ct4_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct4_m,
          "precision": 32,
          "degree_layout": (_ct4_r, _ct4_c),
      },
      {"moduli": list(_ct4_moduli)[:_ct4_m]},
  )
  ct4.polynomial = _ct4_data.reshape(
      _ct4_data.shape[0], _ct4_data.shape[1], _ct4_r, _ct4_c, _ct4_m_in
  )[..., :_ct4_m].copy()
  ct4.batch = ct4.polynomial.shape[0]
  ct4.num_elements = ct4.polynomial.shape[1]
  ct4.num_moduli = _ct4_m
  ct4.degree_layout = (_ct4_r, _ct4_c)
  ct4.r = _ct4_r
  ct4.c = _ct4_c
  ct4.moduli = list(_ct4_moduli)[:_ct4_m]
  ct4.moduli_array = jnp.array(
      ct4.moduli, dtype=getattr(ct4, "modulus_dtype", jnp.uint32)
  )
  _ct5_arg_data = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  _ct5_arg_m_in = _ct5_arg_data.shape[-1]
  _ct5_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct5_arg_m_in
  )
  _ct5_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct5_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct5_arg_r)
  )
  _ct5_arg_moduli = getattr(ct4, "moduli", v0.q_towers)
  if isinstance(_ct5_arg_moduli, (int, np.integer)):
    _ct5_arg_moduli = [int(_ct5_arg_moduli)]
  ct5_arg = Polynomial(
      {
          "batch": _ct5_arg_data.shape[0],
          "num_elements": _ct5_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct5_arg_m,
          "precision": 32,
          "degree_layout": (_ct5_arg_r, _ct5_arg_c),
      },
      {"moduli": list(_ct5_arg_moduli)[:_ct5_arg_m]},
  )
  ct5_arg.polynomial = _ct5_arg_data.reshape(
      _ct5_arg_data.shape[0],
      _ct5_arg_data.shape[1],
      _ct5_arg_r,
      _ct5_arg_c,
      _ct5_arg_m_in,
  )[..., :_ct5_arg_m].copy()
  ct5_arg.batch = ct5_arg.polynomial.shape[0]
  ct5_arg.num_elements = ct5_arg.polynomial.shape[1]
  ct5_arg.num_moduli = _ct5_arg_m
  ct5_arg.degree_layout = (_ct5_arg_r, _ct5_arg_c)
  ct5_arg.r = _ct5_arg_r
  ct5_arg.c = _ct5_arg_c
  ct5_arg.moduli = list(_ct5_arg_moduli)[:_ct5_arg_m]
  ct5_arg.moduli_array = jnp.array(
      ct5_arg.moduli, dtype=getattr(ct5_arg, "modulus_dtype", jnp.uint32)
  )
  ct5_pt_ntt = (
      pt2.polynomial[0, 0, :, : ct5_arg.polynomial.shape[-1]]
      .reshape(ct5_arg.r, ct5_arg.c, ct5_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5_raw = ct5_ptct.mul(ct5_arg, use_bat=False)
  _ct5_data = ct5_raw.polynomial if hasattr(ct5_raw, "polynomial") else ct5_raw
  _ct5_m_in = _ct5_data.shape[-1]
  _ct5_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct5_m_in
  )
  _ct5_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct5_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct5_r)
  )
  _ct5_moduli = getattr(ct5_raw, "moduli", v0.q_towers)
  if isinstance(_ct5_moduli, (int, np.integer)):
    _ct5_moduli = [int(_ct5_moduli)]
  ct5 = Polynomial(
      {
          "batch": _ct5_data.shape[0],
          "num_elements": _ct5_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct5_m,
          "precision": 32,
          "degree_layout": (_ct5_r, _ct5_c),
      },
      {"moduli": list(_ct5_moduli)[:_ct5_m]},
  )
  ct5.polynomial = _ct5_data.reshape(
      _ct5_data.shape[0], _ct5_data.shape[1], _ct5_r, _ct5_c, _ct5_m_in
  )[..., :_ct5_m].copy()
  ct5.batch = ct5.polynomial.shape[0]
  ct5.num_elements = ct5.polynomial.shape[1]
  ct5.num_moduli = _ct5_m
  ct5.degree_layout = (_ct5_r, _ct5_c)
  ct5.r = _ct5_r
  ct5.c = _ct5_c
  ct5.moduli = list(_ct5_moduli)[:_ct5_m]
  ct5.moduli_array = jnp.array(
      ct5.moduli, dtype=getattr(ct5, "modulus_dtype", jnp.uint32)
  )
  _ct6_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct6_arg_m_in = _ct6_arg_data.shape[-1]
  _ct6_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct6_arg_m_in
  )
  _ct6_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct6_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct6_arg_r)
  )
  _ct6_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct6_arg_moduli, (int, np.integer)):
    _ct6_arg_moduli = [int(_ct6_arg_moduli)]
  ct6_arg = Polynomial(
      {
          "batch": _ct6_arg_data.shape[0],
          "num_elements": _ct6_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct6_arg_m,
          "precision": 32,
          "degree_layout": (_ct6_arg_r, _ct6_arg_c),
      },
      {"moduli": list(_ct6_arg_moduli)[:_ct6_arg_m]},
  )
  ct6_arg.polynomial = _ct6_arg_data.reshape(
      _ct6_arg_data.shape[0],
      _ct6_arg_data.shape[1],
      _ct6_arg_r,
      _ct6_arg_c,
      _ct6_arg_m_in,
  )[..., :_ct6_arg_m].copy()
  ct6_arg.batch = ct6_arg.polynomial.shape[0]
  ct6_arg.num_elements = ct6_arg.polynomial.shape[1]
  ct6_arg.num_moduli = _ct6_arg_m
  ct6_arg.degree_layout = (_ct6_arg_r, _ct6_arg_c)
  ct6_arg.r = _ct6_arg_r
  ct6_arg.c = _ct6_arg_c
  ct6_arg.moduli = list(_ct6_arg_moduli)[:_ct6_arg_m]
  ct6_arg.moduli_array = jnp.array(
      ct6_arg.moduli, dtype=getattr(ct6_arg, "modulus_dtype", jnp.uint32)
  )
  ct6_pt_ntt = (
      pt3.polynomial[0, 0, :, : ct6_arg.polynomial.shape[-1]]
      .reshape(ct6_arg.r, ct6_arg.c, ct6_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct6_ptct = v0.ptct_mul[v0.max_level]
  ct6_ptct.set_plaintext(ct6_pt_ntt)
  ct6_raw = ct6_ptct.mul(ct6_arg, use_bat=False)
  _ct6_data = ct6_raw.polynomial if hasattr(ct6_raw, "polynomial") else ct6_raw
  _ct6_m_in = _ct6_data.shape[-1]
  _ct6_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct6_m_in
  )
  _ct6_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct6_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct6_r)
  )
  _ct6_moduli = getattr(ct6_raw, "moduli", v0.q_towers)
  if isinstance(_ct6_moduli, (int, np.integer)):
    _ct6_moduli = [int(_ct6_moduli)]
  ct6 = Polynomial(
      {
          "batch": _ct6_data.shape[0],
          "num_elements": _ct6_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct6_m,
          "precision": 32,
          "degree_layout": (_ct6_r, _ct6_c),
      },
      {"moduli": list(_ct6_moduli)[:_ct6_m]},
  )
  ct6.polynomial = _ct6_data.reshape(
      _ct6_data.shape[0], _ct6_data.shape[1], _ct6_r, _ct6_c, _ct6_m_in
  )[..., :_ct6_m].copy()
  ct6.batch = ct6.polynomial.shape[0]
  ct6.num_elements = ct6.polynomial.shape[1]
  ct6.num_moduli = _ct6_m
  ct6.degree_layout = (_ct6_r, _ct6_c)
  ct6.r = _ct6_r
  ct6.c = _ct6_c
  ct6.moduli = list(_ct6_moduli)[:_ct6_m]
  ct6.moduli_array = jnp.array(
      ct6.moduli, dtype=getattr(ct6, "modulus_dtype", jnp.uint32)
  )
  _ct7_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct7_arg_m_in = _ct7_arg_data.shape[-1]
  _ct7_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct7_arg_m_in
  )
  _ct7_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct7_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct7_arg_r)
  )
  _ct7_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct7_arg_moduli, (int, np.integer)):
    _ct7_arg_moduli = [int(_ct7_arg_moduli)]
  ct7_arg = Polynomial(
      {
          "batch": _ct7_arg_data.shape[0],
          "num_elements": _ct7_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct7_arg_m,
          "precision": 32,
          "degree_layout": (_ct7_arg_r, _ct7_arg_c),
      },
      {"moduli": list(_ct7_arg_moduli)[:_ct7_arg_m]},
  )
  ct7_arg.polynomial = _ct7_arg_data.reshape(
      _ct7_arg_data.shape[0],
      _ct7_arg_data.shape[1],
      _ct7_arg_r,
      _ct7_arg_c,
      _ct7_arg_m_in,
  )[..., :_ct7_arg_m].copy()
  ct7_arg.batch = ct7_arg.polynomial.shape[0]
  ct7_arg.num_elements = ct7_arg.polynomial.shape[1]
  ct7_arg.num_moduli = _ct7_arg_m
  ct7_arg.degree_layout = (_ct7_arg_r, _ct7_arg_c)
  ct7_arg.r = _ct7_arg_r
  ct7_arg.c = _ct7_arg_c
  ct7_arg.moduli = list(_ct7_arg_moduli)[:_ct7_arg_m]
  ct7_arg.moduli_array = jnp.array(
      ct7_arg.moduli, dtype=getattr(ct7_arg, "modulus_dtype", jnp.uint32)
  )
  ct7_pt_ntt = (
      pt4.polynomial[0, 0, :, : ct7_arg.polynomial.shape[-1]]
      .reshape(ct7_arg.r, ct7_arg.c, ct7_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct7_ptct = v0.ptct_mul[v0.max_level]
  ct7_ptct.set_plaintext(ct7_pt_ntt)
  ct7_raw = ct7_ptct.mul(ct7_arg, use_bat=False)
  _ct7_data = ct7_raw.polynomial if hasattr(ct7_raw, "polynomial") else ct7_raw
  _ct7_m_in = _ct7_data.shape[-1]
  _ct7_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct7_m_in
  )
  _ct7_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct7_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct7_r)
  )
  _ct7_moduli = getattr(ct7_raw, "moduli", v0.q_towers)
  if isinstance(_ct7_moduli, (int, np.integer)):
    _ct7_moduli = [int(_ct7_moduli)]
  ct7 = Polynomial(
      {
          "batch": _ct7_data.shape[0],
          "num_elements": _ct7_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct7_m,
          "precision": 32,
          "degree_layout": (_ct7_r, _ct7_c),
      },
      {"moduli": list(_ct7_moduli)[:_ct7_m]},
  )
  ct7.polynomial = _ct7_data.reshape(
      _ct7_data.shape[0], _ct7_data.shape[1], _ct7_r, _ct7_c, _ct7_m_in
  )[..., :_ct7_m].copy()
  ct7.batch = ct7.polynomial.shape[0]
  ct7.num_elements = ct7.polynomial.shape[1]
  ct7.num_moduli = _ct7_m
  ct7.degree_layout = (_ct7_r, _ct7_c)
  ct7.r = _ct7_r
  ct7.c = _ct7_c
  ct7.moduli = list(_ct7_moduli)[:_ct7_m]
  ct7.moduli_array = jnp.array(
      ct7.moduli, dtype=getattr(ct7, "modulus_dtype", jnp.uint32)
  )
  _ct8_arg_data = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  _ct8_arg_m_in = _ct8_arg_data.shape[-1]
  _ct8_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct8_arg_m_in
  )
  _ct8_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct8_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct8_arg_r)
  )
  _ct8_arg_moduli = getattr(ct4, "moduli", v0.q_towers)
  if isinstance(_ct8_arg_moduli, (int, np.integer)):
    _ct8_arg_moduli = [int(_ct8_arg_moduli)]
  ct8_arg = Polynomial(
      {
          "batch": _ct8_arg_data.shape[0],
          "num_elements": _ct8_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct8_arg_m,
          "precision": 32,
          "degree_layout": (_ct8_arg_r, _ct8_arg_c),
      },
      {"moduli": list(_ct8_arg_moduli)[:_ct8_arg_m]},
  )
  ct8_arg.polynomial = _ct8_arg_data.reshape(
      _ct8_arg_data.shape[0],
      _ct8_arg_data.shape[1],
      _ct8_arg_r,
      _ct8_arg_c,
      _ct8_arg_m_in,
  )[..., :_ct8_arg_m].copy()
  ct8_arg.batch = ct8_arg.polynomial.shape[0]
  ct8_arg.num_elements = ct8_arg.polynomial.shape[1]
  ct8_arg.num_moduli = _ct8_arg_m
  ct8_arg.degree_layout = (_ct8_arg_r, _ct8_arg_c)
  ct8_arg.r = _ct8_arg_r
  ct8_arg.c = _ct8_arg_c
  ct8_arg.moduli = list(_ct8_arg_moduli)[:_ct8_arg_m]
  ct8_arg.moduli_array = jnp.array(
      ct8_arg.moduli, dtype=getattr(ct8_arg, "modulus_dtype", jnp.uint32)
  )
  ct8_pt_ntt = (
      pt5.polynomial[0, 0, :, : ct8_arg.polynomial.shape[-1]]
      .reshape(ct8_arg.r, ct8_arg.c, ct8_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct8_ptct = v0.ptct_mul[v0.max_level]
  ct8_ptct.set_plaintext(ct8_pt_ntt)
  ct8_raw = ct8_ptct.mul(ct8_arg, use_bat=False)
  _ct8_data = ct8_raw.polynomial if hasattr(ct8_raw, "polynomial") else ct8_raw
  _ct8_m_in = _ct8_data.shape[-1]
  _ct8_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct8_m_in
  )
  _ct8_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct8_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct8_r)
  )
  _ct8_moduli = getattr(ct8_raw, "moduli", v0.q_towers)
  if isinstance(_ct8_moduli, (int, np.integer)):
    _ct8_moduli = [int(_ct8_moduli)]
  ct8 = Polynomial(
      {
          "batch": _ct8_data.shape[0],
          "num_elements": _ct8_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct8_m,
          "precision": 32,
          "degree_layout": (_ct8_r, _ct8_c),
      },
      {"moduli": list(_ct8_moduli)[:_ct8_m]},
  )
  ct8.polynomial = _ct8_data.reshape(
      _ct8_data.shape[0], _ct8_data.shape[1], _ct8_r, _ct8_c, _ct8_m_in
  )[..., :_ct8_m].copy()
  ct8.batch = ct8.polynomial.shape[0]
  ct8.num_elements = ct8.polynomial.shape[1]
  ct8.num_moduli = _ct8_m
  ct8.degree_layout = (_ct8_r, _ct8_c)
  ct8.r = _ct8_r
  ct8.c = _ct8_c
  ct8.moduli = list(_ct8_moduli)[:_ct8_m]
  ct8.moduli_array = jnp.array(
      ct8.moduli, dtype=getattr(ct8, "modulus_dtype", jnp.uint32)
  )
  _ct9_data = ct6.polynomial if hasattr(ct6, "polynomial") else ct6
  _ct9_m_in = _ct9_data.shape[-1]
  _ct9_m = _ct9_m_in
  _ct9_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct9_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct9_r)
  )
  _ct9_moduli = getattr(ct6, "moduli", v0.q_towers)
  if isinstance(_ct9_moduli, (int, np.integer)):
    _ct9_moduli = [int(_ct9_moduli)]
  ct9 = Polynomial(
      {
          "batch": _ct9_data.shape[0],
          "num_elements": _ct9_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct9_m,
          "precision": 32,
          "degree_layout": (_ct9_r, _ct9_c),
      },
      {"moduli": list(_ct9_moduli)[:_ct9_m]},
  )
  ct9.polynomial = _ct9_data.reshape(
      _ct9_data.shape[0], _ct9_data.shape[1], _ct9_r, _ct9_c, _ct9_m_in
  )[..., :_ct9_m].copy()
  ct9.batch = ct9.polynomial.shape[0]
  ct9.num_elements = ct9.polynomial.shape[1]
  ct9.num_moduli = _ct9_m
  ct9.degree_layout = (_ct9_r, _ct9_c)
  ct9.r = _ct9_r
  ct9.c = _ct9_c
  ct9.moduli = list(_ct9_moduli)[:_ct9_m]
  ct9.moduli_array = jnp.array(
      ct9.moduli, dtype=getattr(ct9, "modulus_dtype", jnp.uint32)
  )
  _ct9_rhs_data = ct7.polynomial if hasattr(ct7, "polynomial") else ct7
  _ct9_rhs_m_in = _ct9_rhs_data.shape[-1]
  _ct9_rhs_m = _ct9_rhs_m_in
  _ct9_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct9_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct9_rhs_r)
  )
  _ct9_rhs_moduli = getattr(ct7, "moduli", v0.q_towers)
  if isinstance(_ct9_rhs_moduli, (int, np.integer)):
    _ct9_rhs_moduli = [int(_ct9_rhs_moduli)]
  ct9_rhs = Polynomial(
      {
          "batch": _ct9_rhs_data.shape[0],
          "num_elements": _ct9_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct9_rhs_m,
          "precision": 32,
          "degree_layout": (_ct9_rhs_r, _ct9_rhs_c),
      },
      {"moduli": list(_ct9_rhs_moduli)[:_ct9_rhs_m]},
  )
  ct9_rhs.polynomial = _ct9_rhs_data.reshape(
      _ct9_rhs_data.shape[0],
      _ct9_rhs_data.shape[1],
      _ct9_rhs_r,
      _ct9_rhs_c,
      _ct9_rhs_m_in,
  )[..., :_ct9_rhs_m].copy()
  ct9_rhs.batch = ct9_rhs.polynomial.shape[0]
  ct9_rhs.num_elements = ct9_rhs.polynomial.shape[1]
  ct9_rhs.num_moduli = _ct9_rhs_m
  ct9_rhs.degree_layout = (_ct9_rhs_r, _ct9_rhs_c)
  ct9_rhs.r = _ct9_rhs_r
  ct9_rhs.c = _ct9_rhs_c
  ct9_rhs.moduli = list(_ct9_rhs_moduli)[:_ct9_rhs_m]
  ct9_rhs.moduli_array = jnp.array(
      ct9_rhs.moduli, dtype=getattr(ct9_rhs, "modulus_dtype", jnp.uint32)
  )
  ct9.add(ct9_rhs)
  _moduli = jnp.array(ct9.moduli, dtype=jnp.uint32)
  ct9.polynomial = jnp.where(
      ct9.polynomial >= _moduli, ct9.polynomial - _moduli, ct9.polynomial
  )
  _ct10_data = ct9.polynomial if hasattr(ct9, "polynomial") else ct9
  _ct10_m_in = _ct10_data.shape[-1]
  _ct10_m = _ct10_m_in
  _ct10_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct10_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct10_r)
  )
  _ct10_moduli = getattr(ct9, "moduli", v0.q_towers)
  if isinstance(_ct10_moduli, (int, np.integer)):
    _ct10_moduli = [int(_ct10_moduli)]
  ct10 = Polynomial(
      {
          "batch": _ct10_data.shape[0],
          "num_elements": _ct10_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct10_m,
          "precision": 32,
          "degree_layout": (_ct10_r, _ct10_c),
      },
      {"moduli": list(_ct10_moduli)[:_ct10_m]},
  )
  ct10.polynomial = _ct10_data.reshape(
      _ct10_data.shape[0], _ct10_data.shape[1], _ct10_r, _ct10_c, _ct10_m_in
  )[..., :_ct10_m].copy()
  ct10.batch = ct10.polynomial.shape[0]
  ct10.num_elements = ct10.polynomial.shape[1]
  ct10.num_moduli = _ct10_m
  ct10.degree_layout = (_ct10_r, _ct10_c)
  ct10.r = _ct10_r
  ct10.c = _ct10_c
  ct10.moduli = list(_ct10_moduli)[:_ct10_m]
  ct10.moduli_array = jnp.array(
      ct10.moduli, dtype=getattr(ct10, "modulus_dtype", jnp.uint32)
  )
  _ct10_rhs_data = ct8.polynomial if hasattr(ct8, "polynomial") else ct8
  _ct10_rhs_m_in = _ct10_rhs_data.shape[-1]
  _ct10_rhs_m = _ct10_rhs_m_in
  _ct10_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct10_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct10_rhs_r)
  )
  _ct10_rhs_moduli = getattr(ct8, "moduli", v0.q_towers)
  if isinstance(_ct10_rhs_moduli, (int, np.integer)):
    _ct10_rhs_moduli = [int(_ct10_rhs_moduli)]
  ct10_rhs = Polynomial(
      {
          "batch": _ct10_rhs_data.shape[0],
          "num_elements": _ct10_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct10_rhs_m,
          "precision": 32,
          "degree_layout": (_ct10_rhs_r, _ct10_rhs_c),
      },
      {"moduli": list(_ct10_rhs_moduli)[:_ct10_rhs_m]},
  )
  ct10_rhs.polynomial = _ct10_rhs_data.reshape(
      _ct10_rhs_data.shape[0],
      _ct10_rhs_data.shape[1],
      _ct10_rhs_r,
      _ct10_rhs_c,
      _ct10_rhs_m_in,
  )[..., :_ct10_rhs_m].copy()
  ct10_rhs.batch = ct10_rhs.polynomial.shape[0]
  ct10_rhs.num_elements = ct10_rhs.polynomial.shape[1]
  ct10_rhs.num_moduli = _ct10_rhs_m
  ct10_rhs.degree_layout = (_ct10_rhs_r, _ct10_rhs_c)
  ct10_rhs.r = _ct10_rhs_r
  ct10_rhs.c = _ct10_rhs_c
  ct10_rhs.moduli = list(_ct10_rhs_moduli)[:_ct10_rhs_m]
  ct10_rhs.moduli_array = jnp.array(
      ct10_rhs.moduli, dtype=getattr(ct10_rhs, "modulus_dtype", jnp.uint32)
  )
  ct10.add(ct10_rhs)
  _moduli = jnp.array(ct10.moduli, dtype=jnp.uint32)
  ct10.polynomial = jnp.where(
      ct10.polynomial >= _moduli, ct10.polynomial - _moduli, ct10.polynomial
  )
  _ct11_arg_data = ct10.polynomial if hasattr(ct10, "polynomial") else ct10
  _ct11_arg_m_in = _ct11_arg_data.shape[-1]
  _ct11_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct11_arg_m_in
  )
  _ct11_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct11_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct11_arg_r)
  )
  _ct11_arg_moduli = getattr(ct10, "moduli", v0.q_towers)
  if isinstance(_ct11_arg_moduli, (int, np.integer)):
    _ct11_arg_moduli = [int(_ct11_arg_moduli)]
  ct11_arg = Polynomial(
      {
          "batch": _ct11_arg_data.shape[0],
          "num_elements": _ct11_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct11_arg_m,
          "precision": 32,
          "degree_layout": (_ct11_arg_r, _ct11_arg_c),
      },
      {"moduli": list(_ct11_arg_moduli)[:_ct11_arg_m]},
  )
  ct11_arg.polynomial = _ct11_arg_data.reshape(
      _ct11_arg_data.shape[0],
      _ct11_arg_data.shape[1],
      _ct11_arg_r,
      _ct11_arg_c,
      _ct11_arg_m_in,
  )[..., :_ct11_arg_m].copy()
  ct11_arg.batch = ct11_arg.polynomial.shape[0]
  ct11_arg.num_elements = ct11_arg.polynomial.shape[1]
  ct11_arg.num_moduli = _ct11_arg_m
  ct11_arg.degree_layout = (_ct11_arg_r, _ct11_arg_c)
  ct11_arg.r = _ct11_arg_r
  ct11_arg.c = _ct11_arg_c
  ct11_arg.moduli = list(_ct11_arg_moduli)[:_ct11_arg_m]
  ct11_arg.moduli_array = jnp.array(
      ct11_arg.moduli, dtype=getattr(ct11_arg, "modulus_dtype", jnp.uint32)
  )
  ct11_raw = v0.he_rot[v0.max_level, 3].rotate(ct11_arg)
  _ct11_data = (
      ct11_raw.polynomial if hasattr(ct11_raw, "polynomial") else ct11_raw
  )
  _ct11_m_in = _ct11_data.shape[-1]
  _ct11_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct11_m_in
  )
  _ct11_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct11_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct11_r)
  )
  _ct11_moduli = getattr(ct11_raw, "moduli", v0.q_towers)
  if isinstance(_ct11_moduli, (int, np.integer)):
    _ct11_moduli = [int(_ct11_moduli)]
  ct11 = Polynomial(
      {
          "batch": _ct11_data.shape[0],
          "num_elements": _ct11_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct11_m,
          "precision": 32,
          "degree_layout": (_ct11_r, _ct11_c),
      },
      {"moduli": list(_ct11_moduli)[:_ct11_m]},
  )
  ct11.polynomial = _ct11_data.reshape(
      _ct11_data.shape[0], _ct11_data.shape[1], _ct11_r, _ct11_c, _ct11_m_in
  )[..., :_ct11_m].copy()
  ct11.batch = ct11.polynomial.shape[0]
  ct11.num_elements = ct11.polynomial.shape[1]
  ct11.num_moduli = _ct11_m
  ct11.degree_layout = (_ct11_r, _ct11_c)
  ct11.r = _ct11_r
  ct11.c = _ct11_c
  ct11.moduli = list(_ct11_moduli)[:_ct11_m]
  ct11.moduli_array = jnp.array(
      ct11.moduli, dtype=getattr(ct11, "modulus_dtype", jnp.uint32)
  )
  _ct12_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct12_arg_m_in = _ct12_arg_data.shape[-1]
  _ct12_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct12_arg_m_in
  )
  _ct12_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct12_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct12_arg_r)
  )
  _ct12_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct12_arg_moduli, (int, np.integer)):
    _ct12_arg_moduli = [int(_ct12_arg_moduli)]
  ct12_arg = Polynomial(
      {
          "batch": _ct12_arg_data.shape[0],
          "num_elements": _ct12_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct12_arg_m,
          "precision": 32,
          "degree_layout": (_ct12_arg_r, _ct12_arg_c),
      },
      {"moduli": list(_ct12_arg_moduli)[:_ct12_arg_m]},
  )
  ct12_arg.polynomial = _ct12_arg_data.reshape(
      _ct12_arg_data.shape[0],
      _ct12_arg_data.shape[1],
      _ct12_arg_r,
      _ct12_arg_c,
      _ct12_arg_m_in,
  )[..., :_ct12_arg_m].copy()
  ct12_arg.batch = ct12_arg.polynomial.shape[0]
  ct12_arg.num_elements = ct12_arg.polynomial.shape[1]
  ct12_arg.num_moduli = _ct12_arg_m
  ct12_arg.degree_layout = (_ct12_arg_r, _ct12_arg_c)
  ct12_arg.r = _ct12_arg_r
  ct12_arg.c = _ct12_arg_c
  ct12_arg.moduli = list(_ct12_arg_moduli)[:_ct12_arg_m]
  ct12_arg.moduli_array = jnp.array(
      ct12_arg.moduli, dtype=getattr(ct12_arg, "modulus_dtype", jnp.uint32)
  )
  ct12_pt_ntt = (
      pt6.polynomial[0, 0, :, : ct12_arg.polynomial.shape[-1]]
      .reshape(ct12_arg.r, ct12_arg.c, ct12_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct12_ptct = v0.ptct_mul[v0.max_level]
  ct12_ptct.set_plaintext(ct12_pt_ntt)
  ct12_raw = ct12_ptct.mul(ct12_arg, use_bat=False)
  _ct12_data = (
      ct12_raw.polynomial if hasattr(ct12_raw, "polynomial") else ct12_raw
  )
  _ct12_m_in = _ct12_data.shape[-1]
  _ct12_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct12_m_in
  )
  _ct12_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct12_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct12_r)
  )
  _ct12_moduli = getattr(ct12_raw, "moduli", v0.q_towers)
  if isinstance(_ct12_moduli, (int, np.integer)):
    _ct12_moduli = [int(_ct12_moduli)]
  ct12 = Polynomial(
      {
          "batch": _ct12_data.shape[0],
          "num_elements": _ct12_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct12_m,
          "precision": 32,
          "degree_layout": (_ct12_r, _ct12_c),
      },
      {"moduli": list(_ct12_moduli)[:_ct12_m]},
  )
  ct12.polynomial = _ct12_data.reshape(
      _ct12_data.shape[0], _ct12_data.shape[1], _ct12_r, _ct12_c, _ct12_m_in
  )[..., :_ct12_m].copy()
  ct12.batch = ct12.polynomial.shape[0]
  ct12.num_elements = ct12.polynomial.shape[1]
  ct12.num_moduli = _ct12_m
  ct12.degree_layout = (_ct12_r, _ct12_c)
  ct12.r = _ct12_r
  ct12.c = _ct12_c
  ct12.moduli = list(_ct12_moduli)[:_ct12_m]
  ct12.moduli_array = jnp.array(
      ct12.moduli, dtype=getattr(ct12, "modulus_dtype", jnp.uint32)
  )
  _ct13_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct13_arg_m_in = _ct13_arg_data.shape[-1]
  _ct13_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct13_arg_m_in
  )
  _ct13_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct13_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct13_arg_r)
  )
  _ct13_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct13_arg_moduli, (int, np.integer)):
    _ct13_arg_moduli = [int(_ct13_arg_moduli)]
  ct13_arg = Polynomial(
      {
          "batch": _ct13_arg_data.shape[0],
          "num_elements": _ct13_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct13_arg_m,
          "precision": 32,
          "degree_layout": (_ct13_arg_r, _ct13_arg_c),
      },
      {"moduli": list(_ct13_arg_moduli)[:_ct13_arg_m]},
  )
  ct13_arg.polynomial = _ct13_arg_data.reshape(
      _ct13_arg_data.shape[0],
      _ct13_arg_data.shape[1],
      _ct13_arg_r,
      _ct13_arg_c,
      _ct13_arg_m_in,
  )[..., :_ct13_arg_m].copy()
  ct13_arg.batch = ct13_arg.polynomial.shape[0]
  ct13_arg.num_elements = ct13_arg.polynomial.shape[1]
  ct13_arg.num_moduli = _ct13_arg_m
  ct13_arg.degree_layout = (_ct13_arg_r, _ct13_arg_c)
  ct13_arg.r = _ct13_arg_r
  ct13_arg.c = _ct13_arg_c
  ct13_arg.moduli = list(_ct13_arg_moduli)[:_ct13_arg_m]
  ct13_arg.moduli_array = jnp.array(
      ct13_arg.moduli, dtype=getattr(ct13_arg, "modulus_dtype", jnp.uint32)
  )
  ct13_pt_ntt = (
      pt7.polynomial[0, 0, :, : ct13_arg.polynomial.shape[-1]]
      .reshape(ct13_arg.r, ct13_arg.c, ct13_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct13_ptct = v0.ptct_mul[v0.max_level]
  ct13_ptct.set_plaintext(ct13_pt_ntt)
  ct13_raw = ct13_ptct.mul(ct13_arg, use_bat=False)
  _ct13_data = (
      ct13_raw.polynomial if hasattr(ct13_raw, "polynomial") else ct13_raw
  )
  _ct13_m_in = _ct13_data.shape[-1]
  _ct13_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct13_m_in
  )
  _ct13_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct13_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct13_r)
  )
  _ct13_moduli = getattr(ct13_raw, "moduli", v0.q_towers)
  if isinstance(_ct13_moduli, (int, np.integer)):
    _ct13_moduli = [int(_ct13_moduli)]
  ct13 = Polynomial(
      {
          "batch": _ct13_data.shape[0],
          "num_elements": _ct13_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct13_m,
          "precision": 32,
          "degree_layout": (_ct13_r, _ct13_c),
      },
      {"moduli": list(_ct13_moduli)[:_ct13_m]},
  )
  ct13.polynomial = _ct13_data.reshape(
      _ct13_data.shape[0], _ct13_data.shape[1], _ct13_r, _ct13_c, _ct13_m_in
  )[..., :_ct13_m].copy()
  ct13.batch = ct13.polynomial.shape[0]
  ct13.num_elements = ct13.polynomial.shape[1]
  ct13.num_moduli = _ct13_m
  ct13.degree_layout = (_ct13_r, _ct13_c)
  ct13.r = _ct13_r
  ct13.c = _ct13_c
  ct13.moduli = list(_ct13_moduli)[:_ct13_m]
  ct13.moduli_array = jnp.array(
      ct13.moduli, dtype=getattr(ct13, "modulus_dtype", jnp.uint32)
  )
  _ct14_data = ct12.polynomial if hasattr(ct12, "polynomial") else ct12
  _ct14_m_in = _ct14_data.shape[-1]
  _ct14_m = _ct14_m_in
  _ct14_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct14_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct14_r)
  )
  _ct14_moduli = getattr(ct12, "moduli", v0.q_towers)
  if isinstance(_ct14_moduli, (int, np.integer)):
    _ct14_moduli = [int(_ct14_moduli)]
  ct14 = Polynomial(
      {
          "batch": _ct14_data.shape[0],
          "num_elements": _ct14_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct14_m,
          "precision": 32,
          "degree_layout": (_ct14_r, _ct14_c),
      },
      {"moduli": list(_ct14_moduli)[:_ct14_m]},
  )
  ct14.polynomial = _ct14_data.reshape(
      _ct14_data.shape[0], _ct14_data.shape[1], _ct14_r, _ct14_c, _ct14_m_in
  )[..., :_ct14_m].copy()
  ct14.batch = ct14.polynomial.shape[0]
  ct14.num_elements = ct14.polynomial.shape[1]
  ct14.num_moduli = _ct14_m
  ct14.degree_layout = (_ct14_r, _ct14_c)
  ct14.r = _ct14_r
  ct14.c = _ct14_c
  ct14.moduli = list(_ct14_moduli)[:_ct14_m]
  ct14.moduli_array = jnp.array(
      ct14.moduli, dtype=getattr(ct14, "modulus_dtype", jnp.uint32)
  )
  _ct14_rhs_data = ct13.polynomial if hasattr(ct13, "polynomial") else ct13
  _ct14_rhs_m_in = _ct14_rhs_data.shape[-1]
  _ct14_rhs_m = _ct14_rhs_m_in
  _ct14_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct14_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct14_rhs_r)
  )
  _ct14_rhs_moduli = getattr(ct13, "moduli", v0.q_towers)
  if isinstance(_ct14_rhs_moduli, (int, np.integer)):
    _ct14_rhs_moduli = [int(_ct14_rhs_moduli)]
  ct14_rhs = Polynomial(
      {
          "batch": _ct14_rhs_data.shape[0],
          "num_elements": _ct14_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct14_rhs_m,
          "precision": 32,
          "degree_layout": (_ct14_rhs_r, _ct14_rhs_c),
      },
      {"moduli": list(_ct14_rhs_moduli)[:_ct14_rhs_m]},
  )
  ct14_rhs.polynomial = _ct14_rhs_data.reshape(
      _ct14_rhs_data.shape[0],
      _ct14_rhs_data.shape[1],
      _ct14_rhs_r,
      _ct14_rhs_c,
      _ct14_rhs_m_in,
  )[..., :_ct14_rhs_m].copy()
  ct14_rhs.batch = ct14_rhs.polynomial.shape[0]
  ct14_rhs.num_elements = ct14_rhs.polynomial.shape[1]
  ct14_rhs.num_moduli = _ct14_rhs_m
  ct14_rhs.degree_layout = (_ct14_rhs_r, _ct14_rhs_c)
  ct14_rhs.r = _ct14_rhs_r
  ct14_rhs.c = _ct14_rhs_c
  ct14_rhs.moduli = list(_ct14_rhs_moduli)[:_ct14_rhs_m]
  ct14_rhs.moduli_array = jnp.array(
      ct14_rhs.moduli, dtype=getattr(ct14_rhs, "modulus_dtype", jnp.uint32)
  )
  ct14.add(ct14_rhs)
  _moduli = jnp.array(ct14.moduli, dtype=jnp.uint32)
  ct14.polynomial = jnp.where(
      ct14.polynomial >= _moduli, ct14.polynomial - _moduli, ct14.polynomial
  )
  _ct15_arg_data = ct14.polynomial if hasattr(ct14, "polynomial") else ct14
  _ct15_arg_m_in = _ct15_arg_data.shape[-1]
  _ct15_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct15_arg_m_in
  )
  _ct15_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct15_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct15_arg_r)
  )
  _ct15_arg_moduli = getattr(ct14, "moduli", v0.q_towers)
  if isinstance(_ct15_arg_moduli, (int, np.integer)):
    _ct15_arg_moduli = [int(_ct15_arg_moduli)]
  ct15_arg = Polynomial(
      {
          "batch": _ct15_arg_data.shape[0],
          "num_elements": _ct15_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct15_arg_m,
          "precision": 32,
          "degree_layout": (_ct15_arg_r, _ct15_arg_c),
      },
      {"moduli": list(_ct15_arg_moduli)[:_ct15_arg_m]},
  )
  ct15_arg.polynomial = _ct15_arg_data.reshape(
      _ct15_arg_data.shape[0],
      _ct15_arg_data.shape[1],
      _ct15_arg_r,
      _ct15_arg_c,
      _ct15_arg_m_in,
  )[..., :_ct15_arg_m].copy()
  ct15_arg.batch = ct15_arg.polynomial.shape[0]
  ct15_arg.num_elements = ct15_arg.polynomial.shape[1]
  ct15_arg.num_moduli = _ct15_arg_m
  ct15_arg.degree_layout = (_ct15_arg_r, _ct15_arg_c)
  ct15_arg.r = _ct15_arg_r
  ct15_arg.c = _ct15_arg_c
  ct15_arg.moduli = list(_ct15_arg_moduli)[:_ct15_arg_m]
  ct15_arg.moduli_array = jnp.array(
      ct15_arg.moduli, dtype=getattr(ct15_arg, "modulus_dtype", jnp.uint32)
  )
  ct15_raw = v0.he_rot[v0.max_level, 6].rotate(ct15_arg)
  _ct15_data = (
      ct15_raw.polynomial if hasattr(ct15_raw, "polynomial") else ct15_raw
  )
  _ct15_m_in = _ct15_data.shape[-1]
  _ct15_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct15_m_in
  )
  _ct15_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct15_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct15_r)
  )
  _ct15_moduli = getattr(ct15_raw, "moduli", v0.q_towers)
  if isinstance(_ct15_moduli, (int, np.integer)):
    _ct15_moduli = [int(_ct15_moduli)]
  ct15 = Polynomial(
      {
          "batch": _ct15_data.shape[0],
          "num_elements": _ct15_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct15_m,
          "precision": 32,
          "degree_layout": (_ct15_r, _ct15_c),
      },
      {"moduli": list(_ct15_moduli)[:_ct15_m]},
  )
  ct15.polynomial = _ct15_data.reshape(
      _ct15_data.shape[0], _ct15_data.shape[1], _ct15_r, _ct15_c, _ct15_m_in
  )[..., :_ct15_m].copy()
  ct15.batch = ct15.polynomial.shape[0]
  ct15.num_elements = ct15.polynomial.shape[1]
  ct15.num_moduli = _ct15_m
  ct15.degree_layout = (_ct15_r, _ct15_c)
  ct15.r = _ct15_r
  ct15.c = _ct15_c
  ct15.moduli = list(_ct15_moduli)[:_ct15_m]
  ct15.moduli_array = jnp.array(
      ct15.moduli, dtype=getattr(ct15, "modulus_dtype", jnp.uint32)
  )
  _ct16_data = ct1.polynomial if hasattr(ct1, "polynomial") else ct1
  _ct16_m_in = _ct16_data.shape[-1]
  _ct16_m = _ct16_m_in
  _ct16_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct16_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct16_r)
  )
  _ct16_moduli = getattr(ct1, "moduli", v0.q_towers)
  if isinstance(_ct16_moduli, (int, np.integer)):
    _ct16_moduli = [int(_ct16_moduli)]
  ct16 = Polynomial(
      {
          "batch": _ct16_data.shape[0],
          "num_elements": _ct16_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct16_m,
          "precision": 32,
          "degree_layout": (_ct16_r, _ct16_c),
      },
      {"moduli": list(_ct16_moduli)[:_ct16_m]},
  )
  ct16.polynomial = _ct16_data.reshape(
      _ct16_data.shape[0], _ct16_data.shape[1], _ct16_r, _ct16_c, _ct16_m_in
  )[..., :_ct16_m].copy()
  ct16.batch = ct16.polynomial.shape[0]
  ct16.num_elements = ct16.polynomial.shape[1]
  ct16.num_moduli = _ct16_m
  ct16.degree_layout = (_ct16_r, _ct16_c)
  ct16.r = _ct16_r
  ct16.c = _ct16_c
  ct16.moduli = list(_ct16_moduli)[:_ct16_m]
  ct16.moduli_array = jnp.array(
      ct16.moduli, dtype=getattr(ct16, "modulus_dtype", jnp.uint32)
  )
  _ct16_rhs_data = ct3.polynomial if hasattr(ct3, "polynomial") else ct3
  _ct16_rhs_m_in = _ct16_rhs_data.shape[-1]
  _ct16_rhs_m = _ct16_rhs_m_in
  _ct16_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct16_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct16_rhs_r)
  )
  _ct16_rhs_moduli = getattr(ct3, "moduli", v0.q_towers)
  if isinstance(_ct16_rhs_moduli, (int, np.integer)):
    _ct16_rhs_moduli = [int(_ct16_rhs_moduli)]
  ct16_rhs = Polynomial(
      {
          "batch": _ct16_rhs_data.shape[0],
          "num_elements": _ct16_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct16_rhs_m,
          "precision": 32,
          "degree_layout": (_ct16_rhs_r, _ct16_rhs_c),
      },
      {"moduli": list(_ct16_rhs_moduli)[:_ct16_rhs_m]},
  )
  ct16_rhs.polynomial = _ct16_rhs_data.reshape(
      _ct16_rhs_data.shape[0],
      _ct16_rhs_data.shape[1],
      _ct16_rhs_r,
      _ct16_rhs_c,
      _ct16_rhs_m_in,
  )[..., :_ct16_rhs_m].copy()
  ct16_rhs.batch = ct16_rhs.polynomial.shape[0]
  ct16_rhs.num_elements = ct16_rhs.polynomial.shape[1]
  ct16_rhs.num_moduli = _ct16_rhs_m
  ct16_rhs.degree_layout = (_ct16_rhs_r, _ct16_rhs_c)
  ct16_rhs.r = _ct16_rhs_r
  ct16_rhs.c = _ct16_rhs_c
  ct16_rhs.moduli = list(_ct16_rhs_moduli)[:_ct16_rhs_m]
  ct16_rhs.moduli_array = jnp.array(
      ct16_rhs.moduli, dtype=getattr(ct16_rhs, "modulus_dtype", jnp.uint32)
  )
  ct16.add(ct16_rhs)
  _moduli = jnp.array(ct16.moduli, dtype=jnp.uint32)
  ct16.polynomial = jnp.where(
      ct16.polynomial >= _moduli, ct16.polynomial - _moduli, ct16.polynomial
  )
  _ct17_data = ct5.polynomial if hasattr(ct5, "polynomial") else ct5
  _ct17_m_in = _ct17_data.shape[-1]
  _ct17_m = _ct17_m_in
  _ct17_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct17_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct17_r)
  )
  _ct17_moduli = getattr(ct5, "moduli", v0.q_towers)
  if isinstance(_ct17_moduli, (int, np.integer)):
    _ct17_moduli = [int(_ct17_moduli)]
  ct17 = Polynomial(
      {
          "batch": _ct17_data.shape[0],
          "num_elements": _ct17_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct17_m,
          "precision": 32,
          "degree_layout": (_ct17_r, _ct17_c),
      },
      {"moduli": list(_ct17_moduli)[:_ct17_m]},
  )
  ct17.polynomial = _ct17_data.reshape(
      _ct17_data.shape[0], _ct17_data.shape[1], _ct17_r, _ct17_c, _ct17_m_in
  )[..., :_ct17_m].copy()
  ct17.batch = ct17.polynomial.shape[0]
  ct17.num_elements = ct17.polynomial.shape[1]
  ct17.num_moduli = _ct17_m
  ct17.degree_layout = (_ct17_r, _ct17_c)
  ct17.r = _ct17_r
  ct17.c = _ct17_c
  ct17.moduli = list(_ct17_moduli)[:_ct17_m]
  ct17.moduli_array = jnp.array(
      ct17.moduli, dtype=getattr(ct17, "modulus_dtype", jnp.uint32)
  )
  _ct17_rhs_data = ct11.polynomial if hasattr(ct11, "polynomial") else ct11
  _ct17_rhs_m_in = _ct17_rhs_data.shape[-1]
  _ct17_rhs_m = _ct17_rhs_m_in
  _ct17_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct17_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct17_rhs_r)
  )
  _ct17_rhs_moduli = getattr(ct11, "moduli", v0.q_towers)
  if isinstance(_ct17_rhs_moduli, (int, np.integer)):
    _ct17_rhs_moduli = [int(_ct17_rhs_moduli)]
  ct17_rhs = Polynomial(
      {
          "batch": _ct17_rhs_data.shape[0],
          "num_elements": _ct17_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct17_rhs_m,
          "precision": 32,
          "degree_layout": (_ct17_rhs_r, _ct17_rhs_c),
      },
      {"moduli": list(_ct17_rhs_moduli)[:_ct17_rhs_m]},
  )
  ct17_rhs.polynomial = _ct17_rhs_data.reshape(
      _ct17_rhs_data.shape[0],
      _ct17_rhs_data.shape[1],
      _ct17_rhs_r,
      _ct17_rhs_c,
      _ct17_rhs_m_in,
  )[..., :_ct17_rhs_m].copy()
  ct17_rhs.batch = ct17_rhs.polynomial.shape[0]
  ct17_rhs.num_elements = ct17_rhs.polynomial.shape[1]
  ct17_rhs.num_moduli = _ct17_rhs_m
  ct17_rhs.degree_layout = (_ct17_rhs_r, _ct17_rhs_c)
  ct17_rhs.r = _ct17_rhs_r
  ct17_rhs.c = _ct17_rhs_c
  ct17_rhs.moduli = list(_ct17_rhs_moduli)[:_ct17_rhs_m]
  ct17_rhs.moduli_array = jnp.array(
      ct17_rhs.moduli, dtype=getattr(ct17_rhs, "modulus_dtype", jnp.uint32)
  )
  ct17.add(ct17_rhs)
  _moduli = jnp.array(ct17.moduli, dtype=jnp.uint32)
  ct17.polynomial = jnp.where(
      ct17.polynomial >= _moduli, ct17.polynomial - _moduli, ct17.polynomial
  )
  _ct18_data = ct17.polynomial if hasattr(ct17, "polynomial") else ct17
  _ct18_m_in = _ct18_data.shape[-1]
  _ct18_m = _ct18_m_in
  _ct18_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct18_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct18_r)
  )
  _ct18_moduli = getattr(ct17, "moduli", v0.q_towers)
  if isinstance(_ct18_moduli, (int, np.integer)):
    _ct18_moduli = [int(_ct18_moduli)]
  ct18 = Polynomial(
      {
          "batch": _ct18_data.shape[0],
          "num_elements": _ct18_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct18_m,
          "precision": 32,
          "degree_layout": (_ct18_r, _ct18_c),
      },
      {"moduli": list(_ct18_moduli)[:_ct18_m]},
  )
  ct18.polynomial = _ct18_data.reshape(
      _ct18_data.shape[0], _ct18_data.shape[1], _ct18_r, _ct18_c, _ct18_m_in
  )[..., :_ct18_m].copy()
  ct18.batch = ct18.polynomial.shape[0]
  ct18.num_elements = ct18.polynomial.shape[1]
  ct18.num_moduli = _ct18_m
  ct18.degree_layout = (_ct18_r, _ct18_c)
  ct18.r = _ct18_r
  ct18.c = _ct18_c
  ct18.moduli = list(_ct18_moduli)[:_ct18_m]
  ct18.moduli_array = jnp.array(
      ct18.moduli, dtype=getattr(ct18, "modulus_dtype", jnp.uint32)
  )
  _ct18_rhs_data = ct15.polynomial if hasattr(ct15, "polynomial") else ct15
  _ct18_rhs_m_in = _ct18_rhs_data.shape[-1]
  _ct18_rhs_m = _ct18_rhs_m_in
  _ct18_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct18_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct18_rhs_r)
  )
  _ct18_rhs_moduli = getattr(ct15, "moduli", v0.q_towers)
  if isinstance(_ct18_rhs_moduli, (int, np.integer)):
    _ct18_rhs_moduli = [int(_ct18_rhs_moduli)]
  ct18_rhs = Polynomial(
      {
          "batch": _ct18_rhs_data.shape[0],
          "num_elements": _ct18_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct18_rhs_m,
          "precision": 32,
          "degree_layout": (_ct18_rhs_r, _ct18_rhs_c),
      },
      {"moduli": list(_ct18_rhs_moduli)[:_ct18_rhs_m]},
  )
  ct18_rhs.polynomial = _ct18_rhs_data.reshape(
      _ct18_rhs_data.shape[0],
      _ct18_rhs_data.shape[1],
      _ct18_rhs_r,
      _ct18_rhs_c,
      _ct18_rhs_m_in,
  )[..., :_ct18_rhs_m].copy()
  ct18_rhs.batch = ct18_rhs.polynomial.shape[0]
  ct18_rhs.num_elements = ct18_rhs.polynomial.shape[1]
  ct18_rhs.num_moduli = _ct18_rhs_m
  ct18_rhs.degree_layout = (_ct18_rhs_r, _ct18_rhs_c)
  ct18_rhs.r = _ct18_rhs_r
  ct18_rhs.c = _ct18_rhs_c
  ct18_rhs.moduli = list(_ct18_rhs_moduli)[:_ct18_rhs_m]
  ct18_rhs.moduli_array = jnp.array(
      ct18_rhs.moduli, dtype=getattr(ct18_rhs, "modulus_dtype", jnp.uint32)
  )
  ct18.add(ct18_rhs)
  _moduli = jnp.array(ct18.moduli, dtype=jnp.uint32)
  ct18.polynomial = jnp.where(
      ct18.polynomial >= _moduli, ct18.polynomial - _moduli, ct18.polynomial
  )
  _ct19_data = ct16.polynomial if hasattr(ct16, "polynomial") else ct16
  _ct19_m_in = _ct19_data.shape[-1]
  _ct19_m = _ct19_m_in
  _ct19_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct19_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct19_r)
  )
  _ct19_moduli = getattr(ct16, "moduli", v0.q_towers)
  if isinstance(_ct19_moduli, (int, np.integer)):
    _ct19_moduli = [int(_ct19_moduli)]
  ct19 = Polynomial(
      {
          "batch": _ct19_data.shape[0],
          "num_elements": _ct19_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct19_m,
          "precision": 32,
          "degree_layout": (_ct19_r, _ct19_c),
      },
      {"moduli": list(_ct19_moduli)[:_ct19_m]},
  )
  ct19.polynomial = _ct19_data.reshape(
      _ct19_data.shape[0], _ct19_data.shape[1], _ct19_r, _ct19_c, _ct19_m_in
  )[..., :_ct19_m].copy()
  ct19.batch = ct19.polynomial.shape[0]
  ct19.num_elements = ct19.polynomial.shape[1]
  ct19.num_moduli = _ct19_m
  ct19.degree_layout = (_ct19_r, _ct19_c)
  ct19.r = _ct19_r
  ct19.c = _ct19_c
  ct19.moduli = list(_ct19_moduli)[:_ct19_m]
  ct19.moduli_array = jnp.array(
      ct19.moduli, dtype=getattr(ct19, "modulus_dtype", jnp.uint32)
  )
  _ct19_rhs_data = ct18.polynomial if hasattr(ct18, "polynomial") else ct18
  _ct19_rhs_m_in = _ct19_rhs_data.shape[-1]
  _ct19_rhs_m = _ct19_rhs_m_in
  _ct19_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct19_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct19_rhs_r)
  )
  _ct19_rhs_moduli = getattr(ct18, "moduli", v0.q_towers)
  if isinstance(_ct19_rhs_moduli, (int, np.integer)):
    _ct19_rhs_moduli = [int(_ct19_rhs_moduli)]
  ct19_rhs = Polynomial(
      {
          "batch": _ct19_rhs_data.shape[0],
          "num_elements": _ct19_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct19_rhs_m,
          "precision": 32,
          "degree_layout": (_ct19_rhs_r, _ct19_rhs_c),
      },
      {"moduli": list(_ct19_rhs_moduli)[:_ct19_rhs_m]},
  )
  ct19_rhs.polynomial = _ct19_rhs_data.reshape(
      _ct19_rhs_data.shape[0],
      _ct19_rhs_data.shape[1],
      _ct19_rhs_r,
      _ct19_rhs_c,
      _ct19_rhs_m_in,
  )[..., :_ct19_rhs_m].copy()
  ct19_rhs.batch = ct19_rhs.polynomial.shape[0]
  ct19_rhs.num_elements = ct19_rhs.polynomial.shape[1]
  ct19_rhs.num_moduli = _ct19_rhs_m
  ct19_rhs.degree_layout = (_ct19_rhs_r, _ct19_rhs_c)
  ct19_rhs.r = _ct19_rhs_r
  ct19_rhs.c = _ct19_rhs_c
  ct19_rhs.moduli = list(_ct19_rhs_moduli)[:_ct19_rhs_m]
  ct19_rhs.moduli_array = jnp.array(
      ct19_rhs.moduli, dtype=getattr(ct19_rhs, "modulus_dtype", jnp.uint32)
  )
  ct19.add(ct19_rhs)
  _moduli = jnp.array(ct19.moduli, dtype=jnp.uint32)
  ct19.polynomial = jnp.where(
      ct19.polynomial >= _moduli, ct19.polynomial - _moduli, ct19.polynomial
  )
  v16 = [None] * 1
  _ct20_arg_data = ct19.polynomial if hasattr(ct19, "polynomial") else ct19
  _ct20_arg_m_in = _ct20_arg_data.shape[-1]
  _ct20_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct20_arg_m_in
  )
  _ct20_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct20_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct20_arg_r)
  )
  _ct20_arg_moduli = getattr(ct19, "moduli", v0.q_towers)
  if isinstance(_ct20_arg_moduli, (int, np.integer)):
    _ct20_arg_moduli = [int(_ct20_arg_moduli)]
  ct20_arg = Polynomial(
      {
          "batch": _ct20_arg_data.shape[0],
          "num_elements": _ct20_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct20_arg_m,
          "precision": 32,
          "degree_layout": (_ct20_arg_r, _ct20_arg_c),
      },
      {"moduli": list(_ct20_arg_moduli)[:_ct20_arg_m]},
  )
  ct20_arg.polynomial = _ct20_arg_data.reshape(
      _ct20_arg_data.shape[0],
      _ct20_arg_data.shape[1],
      _ct20_arg_r,
      _ct20_arg_c,
      _ct20_arg_m_in,
  )[..., :_ct20_arg_m].copy()
  ct20_arg.batch = ct20_arg.polynomial.shape[0]
  ct20_arg.num_elements = ct20_arg.polynomial.shape[1]
  ct20_arg.num_moduli = _ct20_arg_m
  ct20_arg.degree_layout = (_ct20_arg_r, _ct20_arg_c)
  ct20_arg.r = _ct20_arg_r
  ct20_arg.c = _ct20_arg_c
  ct20_arg.moduli = list(_ct20_arg_moduli)[:_ct20_arg_m]
  ct20_arg.moduli_array = jnp.array(
      ct20_arg.moduli, dtype=getattr(ct20_arg, "modulus_dtype", jnp.uint32)
  )
  ct20_raw = v0.he_rescale[v0.max_level, v0.max_level - 1](ct20_arg)
  _ct20_data = (
      ct20_raw.polynomial if hasattr(ct20_raw, "polynomial") else ct20_raw
  )
  _ct20_m_in = _ct20_data.shape[-1]
  _ct20_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct20_m_in
  )
  _ct20_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct20_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct20_r)
  )
  _ct20_moduli = getattr(ct20_raw, "moduli", v0.q_towers)
  if isinstance(_ct20_moduli, (int, np.integer)):
    _ct20_moduli = [int(_ct20_moduli)]
  ct20 = Polynomial(
      {
          "batch": _ct20_data.shape[0],
          "num_elements": _ct20_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct20_m,
          "precision": 32,
          "degree_layout": (_ct20_r, _ct20_c),
      },
      {"moduli": list(_ct20_moduli)[:_ct20_m]},
  )
  ct20.polynomial = _ct20_data.reshape(
      _ct20_data.shape[0], _ct20_data.shape[1], _ct20_r, _ct20_c, _ct20_m_in
  )[..., :_ct20_m].copy()
  ct20.batch = ct20.polynomial.shape[0]
  ct20.num_elements = ct20.polynomial.shape[1]
  ct20.num_moduli = _ct20_m
  ct20.degree_layout = (_ct20_r, _ct20_c)
  ct20.r = _ct20_r
  ct20.c = _ct20_c
  ct20.moduli = list(_ct20_moduli)[:_ct20_m]
  ct20.moduli_array = jnp.array(
      ct20.moduli, dtype=getattr(ct20, "modulus_dtype", jnp.uint32)
  )
  v16[0] = ct20
  v17 = v16
  return v17


def matvec_shift(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  (v3, v4, v5, v6, v7, v8, v9, v10) = matvec_shift__preprocessing(v0, v1)
  v11 = matvec_shift__preprocessed(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)
  return v11


def matvec_shift__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v15)
  v0.public_key = v3
  ct_raw = v0.encrypt(pt)
  _ct_data = ct_raw.polynomial if hasattr(ct_raw, "polynomial") else ct_raw
  _ct_m_in = _ct_data.shape[-1]
  _ct_m = _ct_m_in
  _ct_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct_r)
  )
  _ct_moduli = getattr(ct_raw, "moduli", v0.q_towers)
  if isinstance(_ct_moduli, (int, np.integer)):
    _ct_moduli = [int(_ct_moduli)]
  ct = Polynomial(
      {
          "batch": _ct_data.shape[0],
          "num_elements": _ct_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct_m,
          "precision": 32,
          "degree_layout": (_ct_r, _ct_c),
      },
      {"moduli": list(_ct_moduli)[:_ct_m]},
  )
  ct.polynomial = _ct_data.reshape(
      _ct_data.shape[0], _ct_data.shape[1], _ct_r, _ct_c, _ct_m_in
  )[..., :_ct_m].copy()
  ct.batch = ct.polynomial.shape[0]
  ct.num_elements = ct.polynomial.shape[1]
  ct.num_moduli = _ct_m
  ct.degree_layout = (_ct_r, _ct_c)
  ct.r = _ct_r
  ct.c = _ct_c
  ct.moduli = list(_ct_moduli)[:_ct_m]
  ct.moduli_array = jnp.array(
      ct.moduli, dtype=getattr(ct, "modulus_dtype", jnp.uint32)
  )
  v16 = [ct]
  return v16


def matvec_shift__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 0
  v8 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  _num_moduli = ct.polynomial.shape[-1]
  _q_sub = list(getattr(ct, "moduli", v0.q_towers))[:_num_moduli]
  _ct_for_dec = Polynomial(
      {
          "batch": ct.polynomial.shape[0],
          "num_elements": ct.polynomial.shape[1],
          "degree": v0.degree,
          "precision": 32,
          "num_moduli": _num_moduli,
          "degree_layout": (v0.degree,),
      },
      {"moduli": _q_sub},
  )
  _ct_for_dec.set_batch_polynomial(
      ct.polynomial.reshape(
          ct.polynomial.shape[0], ct.polynomial.shape[1], v0.degree, _num_moduli
      )
  )
  pt = v0.decrypt(_ct_for_dec)
  v9 = v0.decode(pt, is_ntt=False).real.reshape(1, 8)
  v10 = v8.copy()
  for v11 in range(0, 8):
    v13 = int(v11)
    v14 = v9[0, v13]
    v10[v13] = v14
  return v10


def matvec_random__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> (
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
):
  v2 = np.array(
      [
          8.116263e-01,
          1.906357e00,
          1.490788e00,
          1.237451e00,
          3.964354e-01,
          3.963896e-01,
          2.103589e-01,
          1.745735e00,
          1.242118e00,
          1.445338e00,
          1.391105e-01,
          1.942829e00,
          1.681641e00,
          5.034443e-01,
          4.454674e-01,
          4.484686e-01,
          6.780602e-01,
          1.097037e00,
          9.206955e-01,
          6.533354e-01,
          1.262521e00,
          3.650383e-01,
          6.550748e-01,
          7.960875e-01,
          9.665329e-01,
          1.591834e00,
          4.793802e-01,
          1.077045e00,
          1.225588e00,
          1.882558e-01,
          1.254335e00,
          4.239958e-01,
          2.235980e-01,
          1.902883e00,
          1.934701e00,
          1.635955e00,
          6.787661e-01,
          2.855770e-01,
          1.400043e00,
          9.362897e-01,
          3.318726e-01,
          1.040836e00,
          1.653382e-01,
          1.827709e00,
          5.916820e-01,
          1.358792e00,
          6.922510e-01,
          1.088129e00,
          1.138749e00,
          4.512235e-01,
          1.942211e00,
          1.572752e00,
          1.885048e00,
          1.800172e00,
          1.236010e00,
          1.851561e00,
          2.681358e-01,
          4.723674e-01,
          1.859318e-01,
          7.181276e-01,
          8.384869e-01,
          6.155632e-01,
          1.674601e00,
          7.778313e-01,
      ],
      dtype=np.float32,
  ).reshape(8, 8)
  v3 = _assign_layout_15335824159471298539(v2)
  v4 = v3[3 : 3 + 1, 0 : 0 + 5]
  v5 = v3[3 : 3 + 1, 5 : 5 + 3]
  v6 = np.zeros(
      (
          1,
          8,
      ),
      dtype=np.float32,
  )
  v7 = v6.copy()
  v7[0 : 0 + 1, 3 : 3 + 5] = v4
  v8 = v7.copy()
  v8[0 : 0 + 1, 0 : 0 + 3] = v5
  v9 = v3[4 : 4 + 1, 0 : 0 + 5]
  v10 = v3[4 : 4 + 1, 5 : 5 + 3]
  v11 = v6.copy()
  v11[0 : 0 + 1, 3 : 3 + 5] = v9
  v12 = v11.copy()
  v12[0 : 0 + 1, 0 : 0 + 3] = v10
  v13 = v3[5 : 5 + 1, 0 : 0 + 5]
  v14 = v3[5 : 5 + 1, 5 : 5 + 3]
  v15 = v6.copy()
  v15[0 : 0 + 1, 3 : 3 + 5] = v13
  v16 = v15.copy()
  v16[0 : 0 + 1, 0 : 0 + 3] = v14
  v17 = v3[6 : 6 + 1, 0 : 0 + 2]
  v18 = v3[6 : 6 + 1, 2 : 2 + 6]
  v19 = v6.copy()
  v19[0 : 0 + 1, 6 : 6 + 2] = v17
  v20 = v19.copy()
  v20[0 : 0 + 1, 0 : 0 + 6] = v18
  v21 = v3[7 : 7 + 1, 0 : 0 + 2]
  v22 = v3[7 : 7 + 1, 2 : 2 + 6]
  v23 = v6.copy()
  v23[0 : 0 + 1, 6 : 6 + 2] = v21
  v24 = v23.copy()
  v24[0 : 0 + 1, 0 : 0 + 6] = v22
  v25 = v3[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v25)
  v26 = v3[1 : 1 + 1, 0 : 0 + 8].reshape(8)
  pt1 = v0.encode(v26)
  v27 = v3[2 : 2 + 1, 0 : 0 + 8].reshape(8)
  pt2 = v0.encode(v27)
  v28 = v8[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt3 = v0.encode(v28)
  v29 = v12[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt4 = v0.encode(v29)
  v30 = v16[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt5 = v0.encode(v30)
  v31 = v20[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt6 = v0.encode(v31)
  v32 = v24[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt7 = v0.encode(v32)
  v33 = [pt]
  v34 = [pt1]
  v35 = [pt2]
  v36 = [pt3]
  v37 = [pt4]
  v38 = [pt5]
  v39 = [pt6]
  v40 = [pt7]
  return (v33, v34, v35, v36, v37, v38, v39, v40)


def matvec_random__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
    v4: np.ndarray,
    v5: np.ndarray,
    v6: np.ndarray,
    v7: np.ndarray,
    v8: np.ndarray,
    v9: np.ndarray,
    v10: np.ndarray,
) -> np.ndarray:
  v11 = 1
  v12 = 2
  v13 = 3
  v14 = 6
  v15 = 0
  pt = v3[0]
  pt1 = v4[0]
  pt2 = v5[0]
  pt3 = v6[0]
  pt4 = v7[0]
  pt5 = v8[0]
  pt6 = v9[0]
  pt7 = v10[0]
  ct = v2[0]
  _ct1_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct1_arg_m_in = _ct1_arg_data.shape[-1]
  _ct1_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct1_arg_m_in
  )
  _ct1_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct1_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct1_arg_r)
  )
  _ct1_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct1_arg_moduli, (int, np.integer)):
    _ct1_arg_moduli = [int(_ct1_arg_moduli)]
  ct1_arg = Polynomial(
      {
          "batch": _ct1_arg_data.shape[0],
          "num_elements": _ct1_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct1_arg_m,
          "precision": 32,
          "degree_layout": (_ct1_arg_r, _ct1_arg_c),
      },
      {"moduli": list(_ct1_arg_moduli)[:_ct1_arg_m]},
  )
  ct1_arg.polynomial = _ct1_arg_data.reshape(
      _ct1_arg_data.shape[0],
      _ct1_arg_data.shape[1],
      _ct1_arg_r,
      _ct1_arg_c,
      _ct1_arg_m_in,
  )[..., :_ct1_arg_m].copy()
  ct1_arg.batch = ct1_arg.polynomial.shape[0]
  ct1_arg.num_elements = ct1_arg.polynomial.shape[1]
  ct1_arg.num_moduli = _ct1_arg_m
  ct1_arg.degree_layout = (_ct1_arg_r, _ct1_arg_c)
  ct1_arg.r = _ct1_arg_r
  ct1_arg.c = _ct1_arg_c
  ct1_arg.moduli = list(_ct1_arg_moduli)[:_ct1_arg_m]
  ct1_arg.moduli_array = jnp.array(
      ct1_arg.moduli, dtype=getattr(ct1_arg, "modulus_dtype", jnp.uint32)
  )
  ct1_pt_ntt = (
      pt.polynomial[0, 0, :, : ct1_arg.polynomial.shape[-1]]
      .reshape(ct1_arg.r, ct1_arg.c, ct1_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct1_ptct = v0.ptct_mul[v0.max_level]
  ct1_ptct.set_plaintext(ct1_pt_ntt)
  ct1_raw = ct1_ptct.mul(ct1_arg, use_bat=False)
  _ct1_data = ct1_raw.polynomial if hasattr(ct1_raw, "polynomial") else ct1_raw
  _ct1_m_in = _ct1_data.shape[-1]
  _ct1_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct1_m_in
  )
  _ct1_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct1_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct1_r)
  )
  _ct1_moduli = getattr(ct1_raw, "moduli", v0.q_towers)
  if isinstance(_ct1_moduli, (int, np.integer)):
    _ct1_moduli = [int(_ct1_moduli)]
  ct1 = Polynomial(
      {
          "batch": _ct1_data.shape[0],
          "num_elements": _ct1_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct1_m,
          "precision": 32,
          "degree_layout": (_ct1_r, _ct1_c),
      },
      {"moduli": list(_ct1_moduli)[:_ct1_m]},
  )
  ct1.polynomial = _ct1_data.reshape(
      _ct1_data.shape[0], _ct1_data.shape[1], _ct1_r, _ct1_c, _ct1_m_in
  )[..., :_ct1_m].copy()
  ct1.batch = ct1.polynomial.shape[0]
  ct1.num_elements = ct1.polynomial.shape[1]
  ct1.num_moduli = _ct1_m
  ct1.degree_layout = (_ct1_r, _ct1_c)
  ct1.r = _ct1_r
  ct1.c = _ct1_c
  ct1.moduli = list(_ct1_moduli)[:_ct1_m]
  ct1.moduli_array = jnp.array(
      ct1.moduli, dtype=getattr(ct1, "modulus_dtype", jnp.uint32)
  )
  _ct2_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct2_arg_m_in = _ct2_arg_data.shape[-1]
  _ct2_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct2_arg_m_in
  )
  _ct2_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct2_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct2_arg_r)
  )
  _ct2_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct2_arg_moduli, (int, np.integer)):
    _ct2_arg_moduli = [int(_ct2_arg_moduli)]
  ct2_arg = Polynomial(
      {
          "batch": _ct2_arg_data.shape[0],
          "num_elements": _ct2_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct2_arg_m,
          "precision": 32,
          "degree_layout": (_ct2_arg_r, _ct2_arg_c),
      },
      {"moduli": list(_ct2_arg_moduli)[:_ct2_arg_m]},
  )
  ct2_arg.polynomial = _ct2_arg_data.reshape(
      _ct2_arg_data.shape[0],
      _ct2_arg_data.shape[1],
      _ct2_arg_r,
      _ct2_arg_c,
      _ct2_arg_m_in,
  )[..., :_ct2_arg_m].copy()
  ct2_arg.batch = ct2_arg.polynomial.shape[0]
  ct2_arg.num_elements = ct2_arg.polynomial.shape[1]
  ct2_arg.num_moduli = _ct2_arg_m
  ct2_arg.degree_layout = (_ct2_arg_r, _ct2_arg_c)
  ct2_arg.r = _ct2_arg_r
  ct2_arg.c = _ct2_arg_c
  ct2_arg.moduli = list(_ct2_arg_moduli)[:_ct2_arg_m]
  ct2_arg.moduli_array = jnp.array(
      ct2_arg.moduli, dtype=getattr(ct2_arg, "modulus_dtype", jnp.uint32)
  )
  ct2_raw = v0.he_rot[v0.max_level, 1].rotate(ct2_arg)
  _ct2_data = ct2_raw.polynomial if hasattr(ct2_raw, "polynomial") else ct2_raw
  _ct2_m_in = _ct2_data.shape[-1]
  _ct2_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct2_m_in
  )
  _ct2_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct2_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct2_r)
  )
  _ct2_moduli = getattr(ct2_raw, "moduli", v0.q_towers)
  if isinstance(_ct2_moduli, (int, np.integer)):
    _ct2_moduli = [int(_ct2_moduli)]
  ct2 = Polynomial(
      {
          "batch": _ct2_data.shape[0],
          "num_elements": _ct2_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct2_m,
          "precision": 32,
          "degree_layout": (_ct2_r, _ct2_c),
      },
      {"moduli": list(_ct2_moduli)[:_ct2_m]},
  )
  ct2.polynomial = _ct2_data.reshape(
      _ct2_data.shape[0], _ct2_data.shape[1], _ct2_r, _ct2_c, _ct2_m_in
  )[..., :_ct2_m].copy()
  ct2.batch = ct2.polynomial.shape[0]
  ct2.num_elements = ct2.polynomial.shape[1]
  ct2.num_moduli = _ct2_m
  ct2.degree_layout = (_ct2_r, _ct2_c)
  ct2.r = _ct2_r
  ct2.c = _ct2_c
  ct2.moduli = list(_ct2_moduli)[:_ct2_m]
  ct2.moduli_array = jnp.array(
      ct2.moduli, dtype=getattr(ct2, "modulus_dtype", jnp.uint32)
  )
  _ct3_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct3_arg_m_in = _ct3_arg_data.shape[-1]
  _ct3_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct3_arg_m_in
  )
  _ct3_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct3_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct3_arg_r)
  )
  _ct3_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct3_arg_moduli, (int, np.integer)):
    _ct3_arg_moduli = [int(_ct3_arg_moduli)]
  ct3_arg = Polynomial(
      {
          "batch": _ct3_arg_data.shape[0],
          "num_elements": _ct3_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct3_arg_m,
          "precision": 32,
          "degree_layout": (_ct3_arg_r, _ct3_arg_c),
      },
      {"moduli": list(_ct3_arg_moduli)[:_ct3_arg_m]},
  )
  ct3_arg.polynomial = _ct3_arg_data.reshape(
      _ct3_arg_data.shape[0],
      _ct3_arg_data.shape[1],
      _ct3_arg_r,
      _ct3_arg_c,
      _ct3_arg_m_in,
  )[..., :_ct3_arg_m].copy()
  ct3_arg.batch = ct3_arg.polynomial.shape[0]
  ct3_arg.num_elements = ct3_arg.polynomial.shape[1]
  ct3_arg.num_moduli = _ct3_arg_m
  ct3_arg.degree_layout = (_ct3_arg_r, _ct3_arg_c)
  ct3_arg.r = _ct3_arg_r
  ct3_arg.c = _ct3_arg_c
  ct3_arg.moduli = list(_ct3_arg_moduli)[:_ct3_arg_m]
  ct3_arg.moduli_array = jnp.array(
      ct3_arg.moduli, dtype=getattr(ct3_arg, "modulus_dtype", jnp.uint32)
  )
  ct3_pt_ntt = (
      pt1.polynomial[0, 0, :, : ct3_arg.polynomial.shape[-1]]
      .reshape(ct3_arg.r, ct3_arg.c, ct3_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct3_ptct = v0.ptct_mul[v0.max_level]
  ct3_ptct.set_plaintext(ct3_pt_ntt)
  ct3_raw = ct3_ptct.mul(ct3_arg, use_bat=False)
  _ct3_data = ct3_raw.polynomial if hasattr(ct3_raw, "polynomial") else ct3_raw
  _ct3_m_in = _ct3_data.shape[-1]
  _ct3_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct3_m_in
  )
  _ct3_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct3_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct3_r)
  )
  _ct3_moduli = getattr(ct3_raw, "moduli", v0.q_towers)
  if isinstance(_ct3_moduli, (int, np.integer)):
    _ct3_moduli = [int(_ct3_moduli)]
  ct3 = Polynomial(
      {
          "batch": _ct3_data.shape[0],
          "num_elements": _ct3_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct3_m,
          "precision": 32,
          "degree_layout": (_ct3_r, _ct3_c),
      },
      {"moduli": list(_ct3_moduli)[:_ct3_m]},
  )
  ct3.polynomial = _ct3_data.reshape(
      _ct3_data.shape[0], _ct3_data.shape[1], _ct3_r, _ct3_c, _ct3_m_in
  )[..., :_ct3_m].copy()
  ct3.batch = ct3.polynomial.shape[0]
  ct3.num_elements = ct3.polynomial.shape[1]
  ct3.num_moduli = _ct3_m
  ct3.degree_layout = (_ct3_r, _ct3_c)
  ct3.r = _ct3_r
  ct3.c = _ct3_c
  ct3.moduli = list(_ct3_moduli)[:_ct3_m]
  ct3.moduli_array = jnp.array(
      ct3.moduli, dtype=getattr(ct3, "modulus_dtype", jnp.uint32)
  )
  _ct4_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct4_arg_m_in = _ct4_arg_data.shape[-1]
  _ct4_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct4_arg_m_in
  )
  _ct4_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct4_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct4_arg_r)
  )
  _ct4_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct4_arg_moduli, (int, np.integer)):
    _ct4_arg_moduli = [int(_ct4_arg_moduli)]
  ct4_arg = Polynomial(
      {
          "batch": _ct4_arg_data.shape[0],
          "num_elements": _ct4_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct4_arg_m,
          "precision": 32,
          "degree_layout": (_ct4_arg_r, _ct4_arg_c),
      },
      {"moduli": list(_ct4_arg_moduli)[:_ct4_arg_m]},
  )
  ct4_arg.polynomial = _ct4_arg_data.reshape(
      _ct4_arg_data.shape[0],
      _ct4_arg_data.shape[1],
      _ct4_arg_r,
      _ct4_arg_c,
      _ct4_arg_m_in,
  )[..., :_ct4_arg_m].copy()
  ct4_arg.batch = ct4_arg.polynomial.shape[0]
  ct4_arg.num_elements = ct4_arg.polynomial.shape[1]
  ct4_arg.num_moduli = _ct4_arg_m
  ct4_arg.degree_layout = (_ct4_arg_r, _ct4_arg_c)
  ct4_arg.r = _ct4_arg_r
  ct4_arg.c = _ct4_arg_c
  ct4_arg.moduli = list(_ct4_arg_moduli)[:_ct4_arg_m]
  ct4_arg.moduli_array = jnp.array(
      ct4_arg.moduli, dtype=getattr(ct4_arg, "modulus_dtype", jnp.uint32)
  )
  ct4_raw = v0.he_rot[v0.max_level, 2].rotate(ct4_arg)
  _ct4_data = ct4_raw.polynomial if hasattr(ct4_raw, "polynomial") else ct4_raw
  _ct4_m_in = _ct4_data.shape[-1]
  _ct4_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct4_m_in
  )
  _ct4_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct4_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct4_r)
  )
  _ct4_moduli = getattr(ct4_raw, "moduli", v0.q_towers)
  if isinstance(_ct4_moduli, (int, np.integer)):
    _ct4_moduli = [int(_ct4_moduli)]
  ct4 = Polynomial(
      {
          "batch": _ct4_data.shape[0],
          "num_elements": _ct4_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct4_m,
          "precision": 32,
          "degree_layout": (_ct4_r, _ct4_c),
      },
      {"moduli": list(_ct4_moduli)[:_ct4_m]},
  )
  ct4.polynomial = _ct4_data.reshape(
      _ct4_data.shape[0], _ct4_data.shape[1], _ct4_r, _ct4_c, _ct4_m_in
  )[..., :_ct4_m].copy()
  ct4.batch = ct4.polynomial.shape[0]
  ct4.num_elements = ct4.polynomial.shape[1]
  ct4.num_moduli = _ct4_m
  ct4.degree_layout = (_ct4_r, _ct4_c)
  ct4.r = _ct4_r
  ct4.c = _ct4_c
  ct4.moduli = list(_ct4_moduli)[:_ct4_m]
  ct4.moduli_array = jnp.array(
      ct4.moduli, dtype=getattr(ct4, "modulus_dtype", jnp.uint32)
  )
  _ct5_arg_data = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  _ct5_arg_m_in = _ct5_arg_data.shape[-1]
  _ct5_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct5_arg_m_in
  )
  _ct5_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct5_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct5_arg_r)
  )
  _ct5_arg_moduli = getattr(ct4, "moduli", v0.q_towers)
  if isinstance(_ct5_arg_moduli, (int, np.integer)):
    _ct5_arg_moduli = [int(_ct5_arg_moduli)]
  ct5_arg = Polynomial(
      {
          "batch": _ct5_arg_data.shape[0],
          "num_elements": _ct5_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct5_arg_m,
          "precision": 32,
          "degree_layout": (_ct5_arg_r, _ct5_arg_c),
      },
      {"moduli": list(_ct5_arg_moduli)[:_ct5_arg_m]},
  )
  ct5_arg.polynomial = _ct5_arg_data.reshape(
      _ct5_arg_data.shape[0],
      _ct5_arg_data.shape[1],
      _ct5_arg_r,
      _ct5_arg_c,
      _ct5_arg_m_in,
  )[..., :_ct5_arg_m].copy()
  ct5_arg.batch = ct5_arg.polynomial.shape[0]
  ct5_arg.num_elements = ct5_arg.polynomial.shape[1]
  ct5_arg.num_moduli = _ct5_arg_m
  ct5_arg.degree_layout = (_ct5_arg_r, _ct5_arg_c)
  ct5_arg.r = _ct5_arg_r
  ct5_arg.c = _ct5_arg_c
  ct5_arg.moduli = list(_ct5_arg_moduli)[:_ct5_arg_m]
  ct5_arg.moduli_array = jnp.array(
      ct5_arg.moduli, dtype=getattr(ct5_arg, "modulus_dtype", jnp.uint32)
  )
  ct5_pt_ntt = (
      pt2.polynomial[0, 0, :, : ct5_arg.polynomial.shape[-1]]
      .reshape(ct5_arg.r, ct5_arg.c, ct5_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5_raw = ct5_ptct.mul(ct5_arg, use_bat=False)
  _ct5_data = ct5_raw.polynomial if hasattr(ct5_raw, "polynomial") else ct5_raw
  _ct5_m_in = _ct5_data.shape[-1]
  _ct5_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct5_m_in
  )
  _ct5_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct5_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct5_r)
  )
  _ct5_moduli = getattr(ct5_raw, "moduli", v0.q_towers)
  if isinstance(_ct5_moduli, (int, np.integer)):
    _ct5_moduli = [int(_ct5_moduli)]
  ct5 = Polynomial(
      {
          "batch": _ct5_data.shape[0],
          "num_elements": _ct5_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct5_m,
          "precision": 32,
          "degree_layout": (_ct5_r, _ct5_c),
      },
      {"moduli": list(_ct5_moduli)[:_ct5_m]},
  )
  ct5.polynomial = _ct5_data.reshape(
      _ct5_data.shape[0], _ct5_data.shape[1], _ct5_r, _ct5_c, _ct5_m_in
  )[..., :_ct5_m].copy()
  ct5.batch = ct5.polynomial.shape[0]
  ct5.num_elements = ct5.polynomial.shape[1]
  ct5.num_moduli = _ct5_m
  ct5.degree_layout = (_ct5_r, _ct5_c)
  ct5.r = _ct5_r
  ct5.c = _ct5_c
  ct5.moduli = list(_ct5_moduli)[:_ct5_m]
  ct5.moduli_array = jnp.array(
      ct5.moduli, dtype=getattr(ct5, "modulus_dtype", jnp.uint32)
  )
  _ct6_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct6_arg_m_in = _ct6_arg_data.shape[-1]
  _ct6_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct6_arg_m_in
  )
  _ct6_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct6_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct6_arg_r)
  )
  _ct6_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct6_arg_moduli, (int, np.integer)):
    _ct6_arg_moduli = [int(_ct6_arg_moduli)]
  ct6_arg = Polynomial(
      {
          "batch": _ct6_arg_data.shape[0],
          "num_elements": _ct6_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct6_arg_m,
          "precision": 32,
          "degree_layout": (_ct6_arg_r, _ct6_arg_c),
      },
      {"moduli": list(_ct6_arg_moduli)[:_ct6_arg_m]},
  )
  ct6_arg.polynomial = _ct6_arg_data.reshape(
      _ct6_arg_data.shape[0],
      _ct6_arg_data.shape[1],
      _ct6_arg_r,
      _ct6_arg_c,
      _ct6_arg_m_in,
  )[..., :_ct6_arg_m].copy()
  ct6_arg.batch = ct6_arg.polynomial.shape[0]
  ct6_arg.num_elements = ct6_arg.polynomial.shape[1]
  ct6_arg.num_moduli = _ct6_arg_m
  ct6_arg.degree_layout = (_ct6_arg_r, _ct6_arg_c)
  ct6_arg.r = _ct6_arg_r
  ct6_arg.c = _ct6_arg_c
  ct6_arg.moduli = list(_ct6_arg_moduli)[:_ct6_arg_m]
  ct6_arg.moduli_array = jnp.array(
      ct6_arg.moduli, dtype=getattr(ct6_arg, "modulus_dtype", jnp.uint32)
  )
  ct6_pt_ntt = (
      pt3.polynomial[0, 0, :, : ct6_arg.polynomial.shape[-1]]
      .reshape(ct6_arg.r, ct6_arg.c, ct6_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct6_ptct = v0.ptct_mul[v0.max_level]
  ct6_ptct.set_plaintext(ct6_pt_ntt)
  ct6_raw = ct6_ptct.mul(ct6_arg, use_bat=False)
  _ct6_data = ct6_raw.polynomial if hasattr(ct6_raw, "polynomial") else ct6_raw
  _ct6_m_in = _ct6_data.shape[-1]
  _ct6_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct6_m_in
  )
  _ct6_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct6_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct6_r)
  )
  _ct6_moduli = getattr(ct6_raw, "moduli", v0.q_towers)
  if isinstance(_ct6_moduli, (int, np.integer)):
    _ct6_moduli = [int(_ct6_moduli)]
  ct6 = Polynomial(
      {
          "batch": _ct6_data.shape[0],
          "num_elements": _ct6_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct6_m,
          "precision": 32,
          "degree_layout": (_ct6_r, _ct6_c),
      },
      {"moduli": list(_ct6_moduli)[:_ct6_m]},
  )
  ct6.polynomial = _ct6_data.reshape(
      _ct6_data.shape[0], _ct6_data.shape[1], _ct6_r, _ct6_c, _ct6_m_in
  )[..., :_ct6_m].copy()
  ct6.batch = ct6.polynomial.shape[0]
  ct6.num_elements = ct6.polynomial.shape[1]
  ct6.num_moduli = _ct6_m
  ct6.degree_layout = (_ct6_r, _ct6_c)
  ct6.r = _ct6_r
  ct6.c = _ct6_c
  ct6.moduli = list(_ct6_moduli)[:_ct6_m]
  ct6.moduli_array = jnp.array(
      ct6.moduli, dtype=getattr(ct6, "modulus_dtype", jnp.uint32)
  )
  _ct7_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct7_arg_m_in = _ct7_arg_data.shape[-1]
  _ct7_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct7_arg_m_in
  )
  _ct7_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct7_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct7_arg_r)
  )
  _ct7_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct7_arg_moduli, (int, np.integer)):
    _ct7_arg_moduli = [int(_ct7_arg_moduli)]
  ct7_arg = Polynomial(
      {
          "batch": _ct7_arg_data.shape[0],
          "num_elements": _ct7_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct7_arg_m,
          "precision": 32,
          "degree_layout": (_ct7_arg_r, _ct7_arg_c),
      },
      {"moduli": list(_ct7_arg_moduli)[:_ct7_arg_m]},
  )
  ct7_arg.polynomial = _ct7_arg_data.reshape(
      _ct7_arg_data.shape[0],
      _ct7_arg_data.shape[1],
      _ct7_arg_r,
      _ct7_arg_c,
      _ct7_arg_m_in,
  )[..., :_ct7_arg_m].copy()
  ct7_arg.batch = ct7_arg.polynomial.shape[0]
  ct7_arg.num_elements = ct7_arg.polynomial.shape[1]
  ct7_arg.num_moduli = _ct7_arg_m
  ct7_arg.degree_layout = (_ct7_arg_r, _ct7_arg_c)
  ct7_arg.r = _ct7_arg_r
  ct7_arg.c = _ct7_arg_c
  ct7_arg.moduli = list(_ct7_arg_moduli)[:_ct7_arg_m]
  ct7_arg.moduli_array = jnp.array(
      ct7_arg.moduli, dtype=getattr(ct7_arg, "modulus_dtype", jnp.uint32)
  )
  ct7_pt_ntt = (
      pt4.polynomial[0, 0, :, : ct7_arg.polynomial.shape[-1]]
      .reshape(ct7_arg.r, ct7_arg.c, ct7_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct7_ptct = v0.ptct_mul[v0.max_level]
  ct7_ptct.set_plaintext(ct7_pt_ntt)
  ct7_raw = ct7_ptct.mul(ct7_arg, use_bat=False)
  _ct7_data = ct7_raw.polynomial if hasattr(ct7_raw, "polynomial") else ct7_raw
  _ct7_m_in = _ct7_data.shape[-1]
  _ct7_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct7_m_in
  )
  _ct7_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct7_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct7_r)
  )
  _ct7_moduli = getattr(ct7_raw, "moduli", v0.q_towers)
  if isinstance(_ct7_moduli, (int, np.integer)):
    _ct7_moduli = [int(_ct7_moduli)]
  ct7 = Polynomial(
      {
          "batch": _ct7_data.shape[0],
          "num_elements": _ct7_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct7_m,
          "precision": 32,
          "degree_layout": (_ct7_r, _ct7_c),
      },
      {"moduli": list(_ct7_moduli)[:_ct7_m]},
  )
  ct7.polynomial = _ct7_data.reshape(
      _ct7_data.shape[0], _ct7_data.shape[1], _ct7_r, _ct7_c, _ct7_m_in
  )[..., :_ct7_m].copy()
  ct7.batch = ct7.polynomial.shape[0]
  ct7.num_elements = ct7.polynomial.shape[1]
  ct7.num_moduli = _ct7_m
  ct7.degree_layout = (_ct7_r, _ct7_c)
  ct7.r = _ct7_r
  ct7.c = _ct7_c
  ct7.moduli = list(_ct7_moduli)[:_ct7_m]
  ct7.moduli_array = jnp.array(
      ct7.moduli, dtype=getattr(ct7, "modulus_dtype", jnp.uint32)
  )
  _ct8_arg_data = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  _ct8_arg_m_in = _ct8_arg_data.shape[-1]
  _ct8_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct8_arg_m_in
  )
  _ct8_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct8_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct8_arg_r)
  )
  _ct8_arg_moduli = getattr(ct4, "moduli", v0.q_towers)
  if isinstance(_ct8_arg_moduli, (int, np.integer)):
    _ct8_arg_moduli = [int(_ct8_arg_moduli)]
  ct8_arg = Polynomial(
      {
          "batch": _ct8_arg_data.shape[0],
          "num_elements": _ct8_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct8_arg_m,
          "precision": 32,
          "degree_layout": (_ct8_arg_r, _ct8_arg_c),
      },
      {"moduli": list(_ct8_arg_moduli)[:_ct8_arg_m]},
  )
  ct8_arg.polynomial = _ct8_arg_data.reshape(
      _ct8_arg_data.shape[0],
      _ct8_arg_data.shape[1],
      _ct8_arg_r,
      _ct8_arg_c,
      _ct8_arg_m_in,
  )[..., :_ct8_arg_m].copy()
  ct8_arg.batch = ct8_arg.polynomial.shape[0]
  ct8_arg.num_elements = ct8_arg.polynomial.shape[1]
  ct8_arg.num_moduli = _ct8_arg_m
  ct8_arg.degree_layout = (_ct8_arg_r, _ct8_arg_c)
  ct8_arg.r = _ct8_arg_r
  ct8_arg.c = _ct8_arg_c
  ct8_arg.moduli = list(_ct8_arg_moduli)[:_ct8_arg_m]
  ct8_arg.moduli_array = jnp.array(
      ct8_arg.moduli, dtype=getattr(ct8_arg, "modulus_dtype", jnp.uint32)
  )
  ct8_pt_ntt = (
      pt5.polynomial[0, 0, :, : ct8_arg.polynomial.shape[-1]]
      .reshape(ct8_arg.r, ct8_arg.c, ct8_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct8_ptct = v0.ptct_mul[v0.max_level]
  ct8_ptct.set_plaintext(ct8_pt_ntt)
  ct8_raw = ct8_ptct.mul(ct8_arg, use_bat=False)
  _ct8_data = ct8_raw.polynomial if hasattr(ct8_raw, "polynomial") else ct8_raw
  _ct8_m_in = _ct8_data.shape[-1]
  _ct8_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct8_m_in
  )
  _ct8_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct8_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct8_r)
  )
  _ct8_moduli = getattr(ct8_raw, "moduli", v0.q_towers)
  if isinstance(_ct8_moduli, (int, np.integer)):
    _ct8_moduli = [int(_ct8_moduli)]
  ct8 = Polynomial(
      {
          "batch": _ct8_data.shape[0],
          "num_elements": _ct8_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct8_m,
          "precision": 32,
          "degree_layout": (_ct8_r, _ct8_c),
      },
      {"moduli": list(_ct8_moduli)[:_ct8_m]},
  )
  ct8.polynomial = _ct8_data.reshape(
      _ct8_data.shape[0], _ct8_data.shape[1], _ct8_r, _ct8_c, _ct8_m_in
  )[..., :_ct8_m].copy()
  ct8.batch = ct8.polynomial.shape[0]
  ct8.num_elements = ct8.polynomial.shape[1]
  ct8.num_moduli = _ct8_m
  ct8.degree_layout = (_ct8_r, _ct8_c)
  ct8.r = _ct8_r
  ct8.c = _ct8_c
  ct8.moduli = list(_ct8_moduli)[:_ct8_m]
  ct8.moduli_array = jnp.array(
      ct8.moduli, dtype=getattr(ct8, "modulus_dtype", jnp.uint32)
  )
  _ct9_data = ct6.polynomial if hasattr(ct6, "polynomial") else ct6
  _ct9_m_in = _ct9_data.shape[-1]
  _ct9_m = _ct9_m_in
  _ct9_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct9_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct9_r)
  )
  _ct9_moduli = getattr(ct6, "moduli", v0.q_towers)
  if isinstance(_ct9_moduli, (int, np.integer)):
    _ct9_moduli = [int(_ct9_moduli)]
  ct9 = Polynomial(
      {
          "batch": _ct9_data.shape[0],
          "num_elements": _ct9_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct9_m,
          "precision": 32,
          "degree_layout": (_ct9_r, _ct9_c),
      },
      {"moduli": list(_ct9_moduli)[:_ct9_m]},
  )
  ct9.polynomial = _ct9_data.reshape(
      _ct9_data.shape[0], _ct9_data.shape[1], _ct9_r, _ct9_c, _ct9_m_in
  )[..., :_ct9_m].copy()
  ct9.batch = ct9.polynomial.shape[0]
  ct9.num_elements = ct9.polynomial.shape[1]
  ct9.num_moduli = _ct9_m
  ct9.degree_layout = (_ct9_r, _ct9_c)
  ct9.r = _ct9_r
  ct9.c = _ct9_c
  ct9.moduli = list(_ct9_moduli)[:_ct9_m]
  ct9.moduli_array = jnp.array(
      ct9.moduli, dtype=getattr(ct9, "modulus_dtype", jnp.uint32)
  )
  _ct9_rhs_data = ct7.polynomial if hasattr(ct7, "polynomial") else ct7
  _ct9_rhs_m_in = _ct9_rhs_data.shape[-1]
  _ct9_rhs_m = _ct9_rhs_m_in
  _ct9_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct9_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct9_rhs_r)
  )
  _ct9_rhs_moduli = getattr(ct7, "moduli", v0.q_towers)
  if isinstance(_ct9_rhs_moduli, (int, np.integer)):
    _ct9_rhs_moduli = [int(_ct9_rhs_moduli)]
  ct9_rhs = Polynomial(
      {
          "batch": _ct9_rhs_data.shape[0],
          "num_elements": _ct9_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct9_rhs_m,
          "precision": 32,
          "degree_layout": (_ct9_rhs_r, _ct9_rhs_c),
      },
      {"moduli": list(_ct9_rhs_moduli)[:_ct9_rhs_m]},
  )
  ct9_rhs.polynomial = _ct9_rhs_data.reshape(
      _ct9_rhs_data.shape[0],
      _ct9_rhs_data.shape[1],
      _ct9_rhs_r,
      _ct9_rhs_c,
      _ct9_rhs_m_in,
  )[..., :_ct9_rhs_m].copy()
  ct9_rhs.batch = ct9_rhs.polynomial.shape[0]
  ct9_rhs.num_elements = ct9_rhs.polynomial.shape[1]
  ct9_rhs.num_moduli = _ct9_rhs_m
  ct9_rhs.degree_layout = (_ct9_rhs_r, _ct9_rhs_c)
  ct9_rhs.r = _ct9_rhs_r
  ct9_rhs.c = _ct9_rhs_c
  ct9_rhs.moduli = list(_ct9_rhs_moduli)[:_ct9_rhs_m]
  ct9_rhs.moduli_array = jnp.array(
      ct9_rhs.moduli, dtype=getattr(ct9_rhs, "modulus_dtype", jnp.uint32)
  )
  ct9.add(ct9_rhs)
  _moduli = jnp.array(ct9.moduli, dtype=jnp.uint32)
  ct9.polynomial = jnp.where(
      ct9.polynomial >= _moduli, ct9.polynomial - _moduli, ct9.polynomial
  )
  _ct10_data = ct9.polynomial if hasattr(ct9, "polynomial") else ct9
  _ct10_m_in = _ct10_data.shape[-1]
  _ct10_m = _ct10_m_in
  _ct10_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct10_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct10_r)
  )
  _ct10_moduli = getattr(ct9, "moduli", v0.q_towers)
  if isinstance(_ct10_moduli, (int, np.integer)):
    _ct10_moduli = [int(_ct10_moduli)]
  ct10 = Polynomial(
      {
          "batch": _ct10_data.shape[0],
          "num_elements": _ct10_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct10_m,
          "precision": 32,
          "degree_layout": (_ct10_r, _ct10_c),
      },
      {"moduli": list(_ct10_moduli)[:_ct10_m]},
  )
  ct10.polynomial = _ct10_data.reshape(
      _ct10_data.shape[0], _ct10_data.shape[1], _ct10_r, _ct10_c, _ct10_m_in
  )[..., :_ct10_m].copy()
  ct10.batch = ct10.polynomial.shape[0]
  ct10.num_elements = ct10.polynomial.shape[1]
  ct10.num_moduli = _ct10_m
  ct10.degree_layout = (_ct10_r, _ct10_c)
  ct10.r = _ct10_r
  ct10.c = _ct10_c
  ct10.moduli = list(_ct10_moduli)[:_ct10_m]
  ct10.moduli_array = jnp.array(
      ct10.moduli, dtype=getattr(ct10, "modulus_dtype", jnp.uint32)
  )
  _ct10_rhs_data = ct8.polynomial if hasattr(ct8, "polynomial") else ct8
  _ct10_rhs_m_in = _ct10_rhs_data.shape[-1]
  _ct10_rhs_m = _ct10_rhs_m_in
  _ct10_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct10_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct10_rhs_r)
  )
  _ct10_rhs_moduli = getattr(ct8, "moduli", v0.q_towers)
  if isinstance(_ct10_rhs_moduli, (int, np.integer)):
    _ct10_rhs_moduli = [int(_ct10_rhs_moduli)]
  ct10_rhs = Polynomial(
      {
          "batch": _ct10_rhs_data.shape[0],
          "num_elements": _ct10_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct10_rhs_m,
          "precision": 32,
          "degree_layout": (_ct10_rhs_r, _ct10_rhs_c),
      },
      {"moduli": list(_ct10_rhs_moduli)[:_ct10_rhs_m]},
  )
  ct10_rhs.polynomial = _ct10_rhs_data.reshape(
      _ct10_rhs_data.shape[0],
      _ct10_rhs_data.shape[1],
      _ct10_rhs_r,
      _ct10_rhs_c,
      _ct10_rhs_m_in,
  )[..., :_ct10_rhs_m].copy()
  ct10_rhs.batch = ct10_rhs.polynomial.shape[0]
  ct10_rhs.num_elements = ct10_rhs.polynomial.shape[1]
  ct10_rhs.num_moduli = _ct10_rhs_m
  ct10_rhs.degree_layout = (_ct10_rhs_r, _ct10_rhs_c)
  ct10_rhs.r = _ct10_rhs_r
  ct10_rhs.c = _ct10_rhs_c
  ct10_rhs.moduli = list(_ct10_rhs_moduli)[:_ct10_rhs_m]
  ct10_rhs.moduli_array = jnp.array(
      ct10_rhs.moduli, dtype=getattr(ct10_rhs, "modulus_dtype", jnp.uint32)
  )
  ct10.add(ct10_rhs)
  _moduli = jnp.array(ct10.moduli, dtype=jnp.uint32)
  ct10.polynomial = jnp.where(
      ct10.polynomial >= _moduli, ct10.polynomial - _moduli, ct10.polynomial
  )
  _ct11_arg_data = ct10.polynomial if hasattr(ct10, "polynomial") else ct10
  _ct11_arg_m_in = _ct11_arg_data.shape[-1]
  _ct11_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct11_arg_m_in
  )
  _ct11_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct11_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct11_arg_r)
  )
  _ct11_arg_moduli = getattr(ct10, "moduli", v0.q_towers)
  if isinstance(_ct11_arg_moduli, (int, np.integer)):
    _ct11_arg_moduli = [int(_ct11_arg_moduli)]
  ct11_arg = Polynomial(
      {
          "batch": _ct11_arg_data.shape[0],
          "num_elements": _ct11_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct11_arg_m,
          "precision": 32,
          "degree_layout": (_ct11_arg_r, _ct11_arg_c),
      },
      {"moduli": list(_ct11_arg_moduli)[:_ct11_arg_m]},
  )
  ct11_arg.polynomial = _ct11_arg_data.reshape(
      _ct11_arg_data.shape[0],
      _ct11_arg_data.shape[1],
      _ct11_arg_r,
      _ct11_arg_c,
      _ct11_arg_m_in,
  )[..., :_ct11_arg_m].copy()
  ct11_arg.batch = ct11_arg.polynomial.shape[0]
  ct11_arg.num_elements = ct11_arg.polynomial.shape[1]
  ct11_arg.num_moduli = _ct11_arg_m
  ct11_arg.degree_layout = (_ct11_arg_r, _ct11_arg_c)
  ct11_arg.r = _ct11_arg_r
  ct11_arg.c = _ct11_arg_c
  ct11_arg.moduli = list(_ct11_arg_moduli)[:_ct11_arg_m]
  ct11_arg.moduli_array = jnp.array(
      ct11_arg.moduli, dtype=getattr(ct11_arg, "modulus_dtype", jnp.uint32)
  )
  ct11_raw = v0.he_rot[v0.max_level, 3].rotate(ct11_arg)
  _ct11_data = (
      ct11_raw.polynomial if hasattr(ct11_raw, "polynomial") else ct11_raw
  )
  _ct11_m_in = _ct11_data.shape[-1]
  _ct11_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct11_m_in
  )
  _ct11_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct11_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct11_r)
  )
  _ct11_moduli = getattr(ct11_raw, "moduli", v0.q_towers)
  if isinstance(_ct11_moduli, (int, np.integer)):
    _ct11_moduli = [int(_ct11_moduli)]
  ct11 = Polynomial(
      {
          "batch": _ct11_data.shape[0],
          "num_elements": _ct11_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct11_m,
          "precision": 32,
          "degree_layout": (_ct11_r, _ct11_c),
      },
      {"moduli": list(_ct11_moduli)[:_ct11_m]},
  )
  ct11.polynomial = _ct11_data.reshape(
      _ct11_data.shape[0], _ct11_data.shape[1], _ct11_r, _ct11_c, _ct11_m_in
  )[..., :_ct11_m].copy()
  ct11.batch = ct11.polynomial.shape[0]
  ct11.num_elements = ct11.polynomial.shape[1]
  ct11.num_moduli = _ct11_m
  ct11.degree_layout = (_ct11_r, _ct11_c)
  ct11.r = _ct11_r
  ct11.c = _ct11_c
  ct11.moduli = list(_ct11_moduli)[:_ct11_m]
  ct11.moduli_array = jnp.array(
      ct11.moduli, dtype=getattr(ct11, "modulus_dtype", jnp.uint32)
  )
  _ct12_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct12_arg_m_in = _ct12_arg_data.shape[-1]
  _ct12_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct12_arg_m_in
  )
  _ct12_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct12_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct12_arg_r)
  )
  _ct12_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct12_arg_moduli, (int, np.integer)):
    _ct12_arg_moduli = [int(_ct12_arg_moduli)]
  ct12_arg = Polynomial(
      {
          "batch": _ct12_arg_data.shape[0],
          "num_elements": _ct12_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct12_arg_m,
          "precision": 32,
          "degree_layout": (_ct12_arg_r, _ct12_arg_c),
      },
      {"moduli": list(_ct12_arg_moduli)[:_ct12_arg_m]},
  )
  ct12_arg.polynomial = _ct12_arg_data.reshape(
      _ct12_arg_data.shape[0],
      _ct12_arg_data.shape[1],
      _ct12_arg_r,
      _ct12_arg_c,
      _ct12_arg_m_in,
  )[..., :_ct12_arg_m].copy()
  ct12_arg.batch = ct12_arg.polynomial.shape[0]
  ct12_arg.num_elements = ct12_arg.polynomial.shape[1]
  ct12_arg.num_moduli = _ct12_arg_m
  ct12_arg.degree_layout = (_ct12_arg_r, _ct12_arg_c)
  ct12_arg.r = _ct12_arg_r
  ct12_arg.c = _ct12_arg_c
  ct12_arg.moduli = list(_ct12_arg_moduli)[:_ct12_arg_m]
  ct12_arg.moduli_array = jnp.array(
      ct12_arg.moduli, dtype=getattr(ct12_arg, "modulus_dtype", jnp.uint32)
  )
  ct12_pt_ntt = (
      pt6.polynomial[0, 0, :, : ct12_arg.polynomial.shape[-1]]
      .reshape(ct12_arg.r, ct12_arg.c, ct12_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct12_ptct = v0.ptct_mul[v0.max_level]
  ct12_ptct.set_plaintext(ct12_pt_ntt)
  ct12_raw = ct12_ptct.mul(ct12_arg, use_bat=False)
  _ct12_data = (
      ct12_raw.polynomial if hasattr(ct12_raw, "polynomial") else ct12_raw
  )
  _ct12_m_in = _ct12_data.shape[-1]
  _ct12_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct12_m_in
  )
  _ct12_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct12_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct12_r)
  )
  _ct12_moduli = getattr(ct12_raw, "moduli", v0.q_towers)
  if isinstance(_ct12_moduli, (int, np.integer)):
    _ct12_moduli = [int(_ct12_moduli)]
  ct12 = Polynomial(
      {
          "batch": _ct12_data.shape[0],
          "num_elements": _ct12_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct12_m,
          "precision": 32,
          "degree_layout": (_ct12_r, _ct12_c),
      },
      {"moduli": list(_ct12_moduli)[:_ct12_m]},
  )
  ct12.polynomial = _ct12_data.reshape(
      _ct12_data.shape[0], _ct12_data.shape[1], _ct12_r, _ct12_c, _ct12_m_in
  )[..., :_ct12_m].copy()
  ct12.batch = ct12.polynomial.shape[0]
  ct12.num_elements = ct12.polynomial.shape[1]
  ct12.num_moduli = _ct12_m
  ct12.degree_layout = (_ct12_r, _ct12_c)
  ct12.r = _ct12_r
  ct12.c = _ct12_c
  ct12.moduli = list(_ct12_moduli)[:_ct12_m]
  ct12.moduli_array = jnp.array(
      ct12.moduli, dtype=getattr(ct12, "modulus_dtype", jnp.uint32)
  )
  _ct13_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct13_arg_m_in = _ct13_arg_data.shape[-1]
  _ct13_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct13_arg_m_in
  )
  _ct13_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct13_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct13_arg_r)
  )
  _ct13_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct13_arg_moduli, (int, np.integer)):
    _ct13_arg_moduli = [int(_ct13_arg_moduli)]
  ct13_arg = Polynomial(
      {
          "batch": _ct13_arg_data.shape[0],
          "num_elements": _ct13_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct13_arg_m,
          "precision": 32,
          "degree_layout": (_ct13_arg_r, _ct13_arg_c),
      },
      {"moduli": list(_ct13_arg_moduli)[:_ct13_arg_m]},
  )
  ct13_arg.polynomial = _ct13_arg_data.reshape(
      _ct13_arg_data.shape[0],
      _ct13_arg_data.shape[1],
      _ct13_arg_r,
      _ct13_arg_c,
      _ct13_arg_m_in,
  )[..., :_ct13_arg_m].copy()
  ct13_arg.batch = ct13_arg.polynomial.shape[0]
  ct13_arg.num_elements = ct13_arg.polynomial.shape[1]
  ct13_arg.num_moduli = _ct13_arg_m
  ct13_arg.degree_layout = (_ct13_arg_r, _ct13_arg_c)
  ct13_arg.r = _ct13_arg_r
  ct13_arg.c = _ct13_arg_c
  ct13_arg.moduli = list(_ct13_arg_moduli)[:_ct13_arg_m]
  ct13_arg.moduli_array = jnp.array(
      ct13_arg.moduli, dtype=getattr(ct13_arg, "modulus_dtype", jnp.uint32)
  )
  ct13_pt_ntt = (
      pt7.polynomial[0, 0, :, : ct13_arg.polynomial.shape[-1]]
      .reshape(ct13_arg.r, ct13_arg.c, ct13_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct13_ptct = v0.ptct_mul[v0.max_level]
  ct13_ptct.set_plaintext(ct13_pt_ntt)
  ct13_raw = ct13_ptct.mul(ct13_arg, use_bat=False)
  _ct13_data = (
      ct13_raw.polynomial if hasattr(ct13_raw, "polynomial") else ct13_raw
  )
  _ct13_m_in = _ct13_data.shape[-1]
  _ct13_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct13_m_in
  )
  _ct13_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct13_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct13_r)
  )
  _ct13_moduli = getattr(ct13_raw, "moduli", v0.q_towers)
  if isinstance(_ct13_moduli, (int, np.integer)):
    _ct13_moduli = [int(_ct13_moduli)]
  ct13 = Polynomial(
      {
          "batch": _ct13_data.shape[0],
          "num_elements": _ct13_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct13_m,
          "precision": 32,
          "degree_layout": (_ct13_r, _ct13_c),
      },
      {"moduli": list(_ct13_moduli)[:_ct13_m]},
  )
  ct13.polynomial = _ct13_data.reshape(
      _ct13_data.shape[0], _ct13_data.shape[1], _ct13_r, _ct13_c, _ct13_m_in
  )[..., :_ct13_m].copy()
  ct13.batch = ct13.polynomial.shape[0]
  ct13.num_elements = ct13.polynomial.shape[1]
  ct13.num_moduli = _ct13_m
  ct13.degree_layout = (_ct13_r, _ct13_c)
  ct13.r = _ct13_r
  ct13.c = _ct13_c
  ct13.moduli = list(_ct13_moduli)[:_ct13_m]
  ct13.moduli_array = jnp.array(
      ct13.moduli, dtype=getattr(ct13, "modulus_dtype", jnp.uint32)
  )
  _ct14_data = ct12.polynomial if hasattr(ct12, "polynomial") else ct12
  _ct14_m_in = _ct14_data.shape[-1]
  _ct14_m = _ct14_m_in
  _ct14_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct14_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct14_r)
  )
  _ct14_moduli = getattr(ct12, "moduli", v0.q_towers)
  if isinstance(_ct14_moduli, (int, np.integer)):
    _ct14_moduli = [int(_ct14_moduli)]
  ct14 = Polynomial(
      {
          "batch": _ct14_data.shape[0],
          "num_elements": _ct14_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct14_m,
          "precision": 32,
          "degree_layout": (_ct14_r, _ct14_c),
      },
      {"moduli": list(_ct14_moduli)[:_ct14_m]},
  )
  ct14.polynomial = _ct14_data.reshape(
      _ct14_data.shape[0], _ct14_data.shape[1], _ct14_r, _ct14_c, _ct14_m_in
  )[..., :_ct14_m].copy()
  ct14.batch = ct14.polynomial.shape[0]
  ct14.num_elements = ct14.polynomial.shape[1]
  ct14.num_moduli = _ct14_m
  ct14.degree_layout = (_ct14_r, _ct14_c)
  ct14.r = _ct14_r
  ct14.c = _ct14_c
  ct14.moduli = list(_ct14_moduli)[:_ct14_m]
  ct14.moduli_array = jnp.array(
      ct14.moduli, dtype=getattr(ct14, "modulus_dtype", jnp.uint32)
  )
  _ct14_rhs_data = ct13.polynomial if hasattr(ct13, "polynomial") else ct13
  _ct14_rhs_m_in = _ct14_rhs_data.shape[-1]
  _ct14_rhs_m = _ct14_rhs_m_in
  _ct14_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct14_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct14_rhs_r)
  )
  _ct14_rhs_moduli = getattr(ct13, "moduli", v0.q_towers)
  if isinstance(_ct14_rhs_moduli, (int, np.integer)):
    _ct14_rhs_moduli = [int(_ct14_rhs_moduli)]
  ct14_rhs = Polynomial(
      {
          "batch": _ct14_rhs_data.shape[0],
          "num_elements": _ct14_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct14_rhs_m,
          "precision": 32,
          "degree_layout": (_ct14_rhs_r, _ct14_rhs_c),
      },
      {"moduli": list(_ct14_rhs_moduli)[:_ct14_rhs_m]},
  )
  ct14_rhs.polynomial = _ct14_rhs_data.reshape(
      _ct14_rhs_data.shape[0],
      _ct14_rhs_data.shape[1],
      _ct14_rhs_r,
      _ct14_rhs_c,
      _ct14_rhs_m_in,
  )[..., :_ct14_rhs_m].copy()
  ct14_rhs.batch = ct14_rhs.polynomial.shape[0]
  ct14_rhs.num_elements = ct14_rhs.polynomial.shape[1]
  ct14_rhs.num_moduli = _ct14_rhs_m
  ct14_rhs.degree_layout = (_ct14_rhs_r, _ct14_rhs_c)
  ct14_rhs.r = _ct14_rhs_r
  ct14_rhs.c = _ct14_rhs_c
  ct14_rhs.moduli = list(_ct14_rhs_moduli)[:_ct14_rhs_m]
  ct14_rhs.moduli_array = jnp.array(
      ct14_rhs.moduli, dtype=getattr(ct14_rhs, "modulus_dtype", jnp.uint32)
  )
  ct14.add(ct14_rhs)
  _moduli = jnp.array(ct14.moduli, dtype=jnp.uint32)
  ct14.polynomial = jnp.where(
      ct14.polynomial >= _moduli, ct14.polynomial - _moduli, ct14.polynomial
  )
  _ct15_arg_data = ct14.polynomial if hasattr(ct14, "polynomial") else ct14
  _ct15_arg_m_in = _ct15_arg_data.shape[-1]
  _ct15_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct15_arg_m_in
  )
  _ct15_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct15_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct15_arg_r)
  )
  _ct15_arg_moduli = getattr(ct14, "moduli", v0.q_towers)
  if isinstance(_ct15_arg_moduli, (int, np.integer)):
    _ct15_arg_moduli = [int(_ct15_arg_moduli)]
  ct15_arg = Polynomial(
      {
          "batch": _ct15_arg_data.shape[0],
          "num_elements": _ct15_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct15_arg_m,
          "precision": 32,
          "degree_layout": (_ct15_arg_r, _ct15_arg_c),
      },
      {"moduli": list(_ct15_arg_moduli)[:_ct15_arg_m]},
  )
  ct15_arg.polynomial = _ct15_arg_data.reshape(
      _ct15_arg_data.shape[0],
      _ct15_arg_data.shape[1],
      _ct15_arg_r,
      _ct15_arg_c,
      _ct15_arg_m_in,
  )[..., :_ct15_arg_m].copy()
  ct15_arg.batch = ct15_arg.polynomial.shape[0]
  ct15_arg.num_elements = ct15_arg.polynomial.shape[1]
  ct15_arg.num_moduli = _ct15_arg_m
  ct15_arg.degree_layout = (_ct15_arg_r, _ct15_arg_c)
  ct15_arg.r = _ct15_arg_r
  ct15_arg.c = _ct15_arg_c
  ct15_arg.moduli = list(_ct15_arg_moduli)[:_ct15_arg_m]
  ct15_arg.moduli_array = jnp.array(
      ct15_arg.moduli, dtype=getattr(ct15_arg, "modulus_dtype", jnp.uint32)
  )
  ct15_raw = v0.he_rot[v0.max_level, 6].rotate(ct15_arg)
  _ct15_data = (
      ct15_raw.polynomial if hasattr(ct15_raw, "polynomial") else ct15_raw
  )
  _ct15_m_in = _ct15_data.shape[-1]
  _ct15_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct15_m_in
  )
  _ct15_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct15_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct15_r)
  )
  _ct15_moduli = getattr(ct15_raw, "moduli", v0.q_towers)
  if isinstance(_ct15_moduli, (int, np.integer)):
    _ct15_moduli = [int(_ct15_moduli)]
  ct15 = Polynomial(
      {
          "batch": _ct15_data.shape[0],
          "num_elements": _ct15_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct15_m,
          "precision": 32,
          "degree_layout": (_ct15_r, _ct15_c),
      },
      {"moduli": list(_ct15_moduli)[:_ct15_m]},
  )
  ct15.polynomial = _ct15_data.reshape(
      _ct15_data.shape[0], _ct15_data.shape[1], _ct15_r, _ct15_c, _ct15_m_in
  )[..., :_ct15_m].copy()
  ct15.batch = ct15.polynomial.shape[0]
  ct15.num_elements = ct15.polynomial.shape[1]
  ct15.num_moduli = _ct15_m
  ct15.degree_layout = (_ct15_r, _ct15_c)
  ct15.r = _ct15_r
  ct15.c = _ct15_c
  ct15.moduli = list(_ct15_moduli)[:_ct15_m]
  ct15.moduli_array = jnp.array(
      ct15.moduli, dtype=getattr(ct15, "modulus_dtype", jnp.uint32)
  )
  _ct16_data = ct1.polynomial if hasattr(ct1, "polynomial") else ct1
  _ct16_m_in = _ct16_data.shape[-1]
  _ct16_m = _ct16_m_in
  _ct16_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct16_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct16_r)
  )
  _ct16_moduli = getattr(ct1, "moduli", v0.q_towers)
  if isinstance(_ct16_moduli, (int, np.integer)):
    _ct16_moduli = [int(_ct16_moduli)]
  ct16 = Polynomial(
      {
          "batch": _ct16_data.shape[0],
          "num_elements": _ct16_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct16_m,
          "precision": 32,
          "degree_layout": (_ct16_r, _ct16_c),
      },
      {"moduli": list(_ct16_moduli)[:_ct16_m]},
  )
  ct16.polynomial = _ct16_data.reshape(
      _ct16_data.shape[0], _ct16_data.shape[1], _ct16_r, _ct16_c, _ct16_m_in
  )[..., :_ct16_m].copy()
  ct16.batch = ct16.polynomial.shape[0]
  ct16.num_elements = ct16.polynomial.shape[1]
  ct16.num_moduli = _ct16_m
  ct16.degree_layout = (_ct16_r, _ct16_c)
  ct16.r = _ct16_r
  ct16.c = _ct16_c
  ct16.moduli = list(_ct16_moduli)[:_ct16_m]
  ct16.moduli_array = jnp.array(
      ct16.moduli, dtype=getattr(ct16, "modulus_dtype", jnp.uint32)
  )
  _ct16_rhs_data = ct3.polynomial if hasattr(ct3, "polynomial") else ct3
  _ct16_rhs_m_in = _ct16_rhs_data.shape[-1]
  _ct16_rhs_m = _ct16_rhs_m_in
  _ct16_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct16_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct16_rhs_r)
  )
  _ct16_rhs_moduli = getattr(ct3, "moduli", v0.q_towers)
  if isinstance(_ct16_rhs_moduli, (int, np.integer)):
    _ct16_rhs_moduli = [int(_ct16_rhs_moduli)]
  ct16_rhs = Polynomial(
      {
          "batch": _ct16_rhs_data.shape[0],
          "num_elements": _ct16_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct16_rhs_m,
          "precision": 32,
          "degree_layout": (_ct16_rhs_r, _ct16_rhs_c),
      },
      {"moduli": list(_ct16_rhs_moduli)[:_ct16_rhs_m]},
  )
  ct16_rhs.polynomial = _ct16_rhs_data.reshape(
      _ct16_rhs_data.shape[0],
      _ct16_rhs_data.shape[1],
      _ct16_rhs_r,
      _ct16_rhs_c,
      _ct16_rhs_m_in,
  )[..., :_ct16_rhs_m].copy()
  ct16_rhs.batch = ct16_rhs.polynomial.shape[0]
  ct16_rhs.num_elements = ct16_rhs.polynomial.shape[1]
  ct16_rhs.num_moduli = _ct16_rhs_m
  ct16_rhs.degree_layout = (_ct16_rhs_r, _ct16_rhs_c)
  ct16_rhs.r = _ct16_rhs_r
  ct16_rhs.c = _ct16_rhs_c
  ct16_rhs.moduli = list(_ct16_rhs_moduli)[:_ct16_rhs_m]
  ct16_rhs.moduli_array = jnp.array(
      ct16_rhs.moduli, dtype=getattr(ct16_rhs, "modulus_dtype", jnp.uint32)
  )
  ct16.add(ct16_rhs)
  _moduli = jnp.array(ct16.moduli, dtype=jnp.uint32)
  ct16.polynomial = jnp.where(
      ct16.polynomial >= _moduli, ct16.polynomial - _moduli, ct16.polynomial
  )
  _ct17_data = ct5.polynomial if hasattr(ct5, "polynomial") else ct5
  _ct17_m_in = _ct17_data.shape[-1]
  _ct17_m = _ct17_m_in
  _ct17_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct17_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct17_r)
  )
  _ct17_moduli = getattr(ct5, "moduli", v0.q_towers)
  if isinstance(_ct17_moduli, (int, np.integer)):
    _ct17_moduli = [int(_ct17_moduli)]
  ct17 = Polynomial(
      {
          "batch": _ct17_data.shape[0],
          "num_elements": _ct17_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct17_m,
          "precision": 32,
          "degree_layout": (_ct17_r, _ct17_c),
      },
      {"moduli": list(_ct17_moduli)[:_ct17_m]},
  )
  ct17.polynomial = _ct17_data.reshape(
      _ct17_data.shape[0], _ct17_data.shape[1], _ct17_r, _ct17_c, _ct17_m_in
  )[..., :_ct17_m].copy()
  ct17.batch = ct17.polynomial.shape[0]
  ct17.num_elements = ct17.polynomial.shape[1]
  ct17.num_moduli = _ct17_m
  ct17.degree_layout = (_ct17_r, _ct17_c)
  ct17.r = _ct17_r
  ct17.c = _ct17_c
  ct17.moduli = list(_ct17_moduli)[:_ct17_m]
  ct17.moduli_array = jnp.array(
      ct17.moduli, dtype=getattr(ct17, "modulus_dtype", jnp.uint32)
  )
  _ct17_rhs_data = ct11.polynomial if hasattr(ct11, "polynomial") else ct11
  _ct17_rhs_m_in = _ct17_rhs_data.shape[-1]
  _ct17_rhs_m = _ct17_rhs_m_in
  _ct17_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct17_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct17_rhs_r)
  )
  _ct17_rhs_moduli = getattr(ct11, "moduli", v0.q_towers)
  if isinstance(_ct17_rhs_moduli, (int, np.integer)):
    _ct17_rhs_moduli = [int(_ct17_rhs_moduli)]
  ct17_rhs = Polynomial(
      {
          "batch": _ct17_rhs_data.shape[0],
          "num_elements": _ct17_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct17_rhs_m,
          "precision": 32,
          "degree_layout": (_ct17_rhs_r, _ct17_rhs_c),
      },
      {"moduli": list(_ct17_rhs_moduli)[:_ct17_rhs_m]},
  )
  ct17_rhs.polynomial = _ct17_rhs_data.reshape(
      _ct17_rhs_data.shape[0],
      _ct17_rhs_data.shape[1],
      _ct17_rhs_r,
      _ct17_rhs_c,
      _ct17_rhs_m_in,
  )[..., :_ct17_rhs_m].copy()
  ct17_rhs.batch = ct17_rhs.polynomial.shape[0]
  ct17_rhs.num_elements = ct17_rhs.polynomial.shape[1]
  ct17_rhs.num_moduli = _ct17_rhs_m
  ct17_rhs.degree_layout = (_ct17_rhs_r, _ct17_rhs_c)
  ct17_rhs.r = _ct17_rhs_r
  ct17_rhs.c = _ct17_rhs_c
  ct17_rhs.moduli = list(_ct17_rhs_moduli)[:_ct17_rhs_m]
  ct17_rhs.moduli_array = jnp.array(
      ct17_rhs.moduli, dtype=getattr(ct17_rhs, "modulus_dtype", jnp.uint32)
  )
  ct17.add(ct17_rhs)
  _moduli = jnp.array(ct17.moduli, dtype=jnp.uint32)
  ct17.polynomial = jnp.where(
      ct17.polynomial >= _moduli, ct17.polynomial - _moduli, ct17.polynomial
  )
  _ct18_data = ct17.polynomial if hasattr(ct17, "polynomial") else ct17
  _ct18_m_in = _ct18_data.shape[-1]
  _ct18_m = _ct18_m_in
  _ct18_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct18_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct18_r)
  )
  _ct18_moduli = getattr(ct17, "moduli", v0.q_towers)
  if isinstance(_ct18_moduli, (int, np.integer)):
    _ct18_moduli = [int(_ct18_moduli)]
  ct18 = Polynomial(
      {
          "batch": _ct18_data.shape[0],
          "num_elements": _ct18_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct18_m,
          "precision": 32,
          "degree_layout": (_ct18_r, _ct18_c),
      },
      {"moduli": list(_ct18_moduli)[:_ct18_m]},
  )
  ct18.polynomial = _ct18_data.reshape(
      _ct18_data.shape[0], _ct18_data.shape[1], _ct18_r, _ct18_c, _ct18_m_in
  )[..., :_ct18_m].copy()
  ct18.batch = ct18.polynomial.shape[0]
  ct18.num_elements = ct18.polynomial.shape[1]
  ct18.num_moduli = _ct18_m
  ct18.degree_layout = (_ct18_r, _ct18_c)
  ct18.r = _ct18_r
  ct18.c = _ct18_c
  ct18.moduli = list(_ct18_moduli)[:_ct18_m]
  ct18.moduli_array = jnp.array(
      ct18.moduli, dtype=getattr(ct18, "modulus_dtype", jnp.uint32)
  )
  _ct18_rhs_data = ct15.polynomial if hasattr(ct15, "polynomial") else ct15
  _ct18_rhs_m_in = _ct18_rhs_data.shape[-1]
  _ct18_rhs_m = _ct18_rhs_m_in
  _ct18_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct18_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct18_rhs_r)
  )
  _ct18_rhs_moduli = getattr(ct15, "moduli", v0.q_towers)
  if isinstance(_ct18_rhs_moduli, (int, np.integer)):
    _ct18_rhs_moduli = [int(_ct18_rhs_moduli)]
  ct18_rhs = Polynomial(
      {
          "batch": _ct18_rhs_data.shape[0],
          "num_elements": _ct18_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct18_rhs_m,
          "precision": 32,
          "degree_layout": (_ct18_rhs_r, _ct18_rhs_c),
      },
      {"moduli": list(_ct18_rhs_moduli)[:_ct18_rhs_m]},
  )
  ct18_rhs.polynomial = _ct18_rhs_data.reshape(
      _ct18_rhs_data.shape[0],
      _ct18_rhs_data.shape[1],
      _ct18_rhs_r,
      _ct18_rhs_c,
      _ct18_rhs_m_in,
  )[..., :_ct18_rhs_m].copy()
  ct18_rhs.batch = ct18_rhs.polynomial.shape[0]
  ct18_rhs.num_elements = ct18_rhs.polynomial.shape[1]
  ct18_rhs.num_moduli = _ct18_rhs_m
  ct18_rhs.degree_layout = (_ct18_rhs_r, _ct18_rhs_c)
  ct18_rhs.r = _ct18_rhs_r
  ct18_rhs.c = _ct18_rhs_c
  ct18_rhs.moduli = list(_ct18_rhs_moduli)[:_ct18_rhs_m]
  ct18_rhs.moduli_array = jnp.array(
      ct18_rhs.moduli, dtype=getattr(ct18_rhs, "modulus_dtype", jnp.uint32)
  )
  ct18.add(ct18_rhs)
  _moduli = jnp.array(ct18.moduli, dtype=jnp.uint32)
  ct18.polynomial = jnp.where(
      ct18.polynomial >= _moduli, ct18.polynomial - _moduli, ct18.polynomial
  )
  _ct19_data = ct16.polynomial if hasattr(ct16, "polynomial") else ct16
  _ct19_m_in = _ct19_data.shape[-1]
  _ct19_m = _ct19_m_in
  _ct19_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct19_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct19_r)
  )
  _ct19_moduli = getattr(ct16, "moduli", v0.q_towers)
  if isinstance(_ct19_moduli, (int, np.integer)):
    _ct19_moduli = [int(_ct19_moduli)]
  ct19 = Polynomial(
      {
          "batch": _ct19_data.shape[0],
          "num_elements": _ct19_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct19_m,
          "precision": 32,
          "degree_layout": (_ct19_r, _ct19_c),
      },
      {"moduli": list(_ct19_moduli)[:_ct19_m]},
  )
  ct19.polynomial = _ct19_data.reshape(
      _ct19_data.shape[0], _ct19_data.shape[1], _ct19_r, _ct19_c, _ct19_m_in
  )[..., :_ct19_m].copy()
  ct19.batch = ct19.polynomial.shape[0]
  ct19.num_elements = ct19.polynomial.shape[1]
  ct19.num_moduli = _ct19_m
  ct19.degree_layout = (_ct19_r, _ct19_c)
  ct19.r = _ct19_r
  ct19.c = _ct19_c
  ct19.moduli = list(_ct19_moduli)[:_ct19_m]
  ct19.moduli_array = jnp.array(
      ct19.moduli, dtype=getattr(ct19, "modulus_dtype", jnp.uint32)
  )
  _ct19_rhs_data = ct18.polynomial if hasattr(ct18, "polynomial") else ct18
  _ct19_rhs_m_in = _ct19_rhs_data.shape[-1]
  _ct19_rhs_m = _ct19_rhs_m_in
  _ct19_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct19_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct19_rhs_r)
  )
  _ct19_rhs_moduli = getattr(ct18, "moduli", v0.q_towers)
  if isinstance(_ct19_rhs_moduli, (int, np.integer)):
    _ct19_rhs_moduli = [int(_ct19_rhs_moduli)]
  ct19_rhs = Polynomial(
      {
          "batch": _ct19_rhs_data.shape[0],
          "num_elements": _ct19_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct19_rhs_m,
          "precision": 32,
          "degree_layout": (_ct19_rhs_r, _ct19_rhs_c),
      },
      {"moduli": list(_ct19_rhs_moduli)[:_ct19_rhs_m]},
  )
  ct19_rhs.polynomial = _ct19_rhs_data.reshape(
      _ct19_rhs_data.shape[0],
      _ct19_rhs_data.shape[1],
      _ct19_rhs_r,
      _ct19_rhs_c,
      _ct19_rhs_m_in,
  )[..., :_ct19_rhs_m].copy()
  ct19_rhs.batch = ct19_rhs.polynomial.shape[0]
  ct19_rhs.num_elements = ct19_rhs.polynomial.shape[1]
  ct19_rhs.num_moduli = _ct19_rhs_m
  ct19_rhs.degree_layout = (_ct19_rhs_r, _ct19_rhs_c)
  ct19_rhs.r = _ct19_rhs_r
  ct19_rhs.c = _ct19_rhs_c
  ct19_rhs.moduli = list(_ct19_rhs_moduli)[:_ct19_rhs_m]
  ct19_rhs.moduli_array = jnp.array(
      ct19_rhs.moduli, dtype=getattr(ct19_rhs, "modulus_dtype", jnp.uint32)
  )
  ct19.add(ct19_rhs)
  _moduli = jnp.array(ct19.moduli, dtype=jnp.uint32)
  ct19.polynomial = jnp.where(
      ct19.polynomial >= _moduli, ct19.polynomial - _moduli, ct19.polynomial
  )
  v16 = [None] * 1
  _ct20_arg_data = ct19.polynomial if hasattr(ct19, "polynomial") else ct19
  _ct20_arg_m_in = _ct20_arg_data.shape[-1]
  _ct20_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct20_arg_m_in
  )
  _ct20_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct20_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct20_arg_r)
  )
  _ct20_arg_moduli = getattr(ct19, "moduli", v0.q_towers)
  if isinstance(_ct20_arg_moduli, (int, np.integer)):
    _ct20_arg_moduli = [int(_ct20_arg_moduli)]
  ct20_arg = Polynomial(
      {
          "batch": _ct20_arg_data.shape[0],
          "num_elements": _ct20_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct20_arg_m,
          "precision": 32,
          "degree_layout": (_ct20_arg_r, _ct20_arg_c),
      },
      {"moduli": list(_ct20_arg_moduli)[:_ct20_arg_m]},
  )
  ct20_arg.polynomial = _ct20_arg_data.reshape(
      _ct20_arg_data.shape[0],
      _ct20_arg_data.shape[1],
      _ct20_arg_r,
      _ct20_arg_c,
      _ct20_arg_m_in,
  )[..., :_ct20_arg_m].copy()
  ct20_arg.batch = ct20_arg.polynomial.shape[0]
  ct20_arg.num_elements = ct20_arg.polynomial.shape[1]
  ct20_arg.num_moduli = _ct20_arg_m
  ct20_arg.degree_layout = (_ct20_arg_r, _ct20_arg_c)
  ct20_arg.r = _ct20_arg_r
  ct20_arg.c = _ct20_arg_c
  ct20_arg.moduli = list(_ct20_arg_moduli)[:_ct20_arg_m]
  ct20_arg.moduli_array = jnp.array(
      ct20_arg.moduli, dtype=getattr(ct20_arg, "modulus_dtype", jnp.uint32)
  )
  ct20_raw = v0.he_rescale[v0.max_level, v0.max_level - 1](ct20_arg)
  _ct20_data = (
      ct20_raw.polynomial if hasattr(ct20_raw, "polynomial") else ct20_raw
  )
  _ct20_m_in = _ct20_data.shape[-1]
  _ct20_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct20_m_in
  )
  _ct20_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct20_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct20_r)
  )
  _ct20_moduli = getattr(ct20_raw, "moduli", v0.q_towers)
  if isinstance(_ct20_moduli, (int, np.integer)):
    _ct20_moduli = [int(_ct20_moduli)]
  ct20 = Polynomial(
      {
          "batch": _ct20_data.shape[0],
          "num_elements": _ct20_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct20_m,
          "precision": 32,
          "degree_layout": (_ct20_r, _ct20_c),
      },
      {"moduli": list(_ct20_moduli)[:_ct20_m]},
  )
  ct20.polynomial = _ct20_data.reshape(
      _ct20_data.shape[0], _ct20_data.shape[1], _ct20_r, _ct20_c, _ct20_m_in
  )[..., :_ct20_m].copy()
  ct20.batch = ct20.polynomial.shape[0]
  ct20.num_elements = ct20.polynomial.shape[1]
  ct20.num_moduli = _ct20_m
  ct20.degree_layout = (_ct20_r, _ct20_c)
  ct20.r = _ct20_r
  ct20.c = _ct20_c
  ct20.moduli = list(_ct20_moduli)[:_ct20_m]
  ct20.moduli_array = jnp.array(
      ct20.moduli, dtype=getattr(ct20, "modulus_dtype", jnp.uint32)
  )
  v16[0] = ct20
  v17 = v16
  return v17


def matvec_random(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  (v3, v4, v5, v6, v7, v8, v9, v10) = matvec_random__preprocessing(v0, v1)
  v11 = matvec_random__preprocessed(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)
  return v11


def matvec_random__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v15)
  v0.public_key = v3
  ct_raw = v0.encrypt(pt)
  _ct_data = ct_raw.polynomial if hasattr(ct_raw, "polynomial") else ct_raw
  _ct_m_in = _ct_data.shape[-1]
  _ct_m = _ct_m_in
  _ct_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct_r)
  )
  _ct_moduli = getattr(ct_raw, "moduli", v0.q_towers)
  if isinstance(_ct_moduli, (int, np.integer)):
    _ct_moduli = [int(_ct_moduli)]
  ct = Polynomial(
      {
          "batch": _ct_data.shape[0],
          "num_elements": _ct_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct_m,
          "precision": 32,
          "degree_layout": (_ct_r, _ct_c),
      },
      {"moduli": list(_ct_moduli)[:_ct_m]},
  )
  ct.polynomial = _ct_data.reshape(
      _ct_data.shape[0], _ct_data.shape[1], _ct_r, _ct_c, _ct_m_in
  )[..., :_ct_m].copy()
  ct.batch = ct.polynomial.shape[0]
  ct.num_elements = ct.polynomial.shape[1]
  ct.num_moduli = _ct_m
  ct.degree_layout = (_ct_r, _ct_c)
  ct.r = _ct_r
  ct.c = _ct_c
  ct.moduli = list(_ct_moduli)[:_ct_m]
  ct.moduli_array = jnp.array(
      ct.moduli, dtype=getattr(ct, "modulus_dtype", jnp.uint32)
  )
  v16 = [ct]
  return v16


def matvec_random__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 0
  v8 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  _num_moduli = ct.polynomial.shape[-1]
  _q_sub = list(getattr(ct, "moduli", v0.q_towers))[:_num_moduli]
  _ct_for_dec = Polynomial(
      {
          "batch": ct.polynomial.shape[0],
          "num_elements": ct.polynomial.shape[1],
          "degree": v0.degree,
          "precision": 32,
          "num_moduli": _num_moduli,
          "degree_layout": (v0.degree,),
      },
      {"moduli": _q_sub},
  )
  _ct_for_dec.set_batch_polynomial(
      ct.polynomial.reshape(
          ct.polynomial.shape[0], ct.polynomial.shape[1], v0.degree, _num_moduli
      )
  )
  pt = v0.decrypt(_ct_for_dec)
  v9 = v0.decode(pt, is_ntt=False).real.reshape(1, 8)
  v10 = v8.copy()
  for v11 in range(0, 8):
    v13 = int(v11)
    v14 = v9[0, v13]
    v10[v13] = v14
  return v10


def matvec_chain__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> (
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
):
  v2 = np.array(
      [
          1.340000e00,
          5.800000e-01,
          1.260000e00,
          7.400000e-01,
          6.900000e-01,
          1.070000e00,
          6.000000e-01,
          1.390000e00,
          1.130000e00,
          1.220000e00,
          5.200000e-01,
          1.090000e00,
          1.060000e00,
          6.600000e-01,
          6.500000e-01,
          1.200000e00,
          8.200000e-01,
          1.190000e00,
          1.050000e00,
          8.900000e-01,
          1.430000e00,
          1.340000e00,
          8.600000e-01,
          5.400000e-01,
          8.000000e-01,
          9.000000e-01,
          1.200000e00,
          1.500000e00,
          8.600000e-01,
          1.260000e00,
          1.090000e00,
          1.190000e00,
          6.500000e-01,
          9.000000e-01,
          7.400000e-01,
          8.400000e-01,
          1.010000e00,
          1.170000e00,
          6.100000e-01,
          6.300000e-01,
          8.200000e-01,
          1.160000e00,
          1.350000e00,
          1.050000e00,
          1.350000e00,
          8.800000e-01,
          8.200000e-01,
          8.500000e-01,
          6.700000e-01,
          1.330000e00,
          8.400000e-01,
          1.050000e00,
          1.080000e00,
          1.020000e00,
          5.000000e-01,
          1.490000e00,
          1.410000e00,
          7.100000e-01,
          7.900000e-01,
          1.020000e00,
          1.400000e00,
          1.480000e00,
          7.600000e-01,
          1.060000e00,
      ],
      dtype=np.float32,
  ).reshape(8, 8)
  v3 = np.array(
      [
          1.200000e00,
          7.900000e-01,
          7.300000e-01,
          1.050000e00,
          1.220000e00,
          9.200000e-01,
          1.480000e00,
          1.180000e00,
          9.800000e-01,
          8.900000e-01,
          8.400000e-01,
          1.230000e00,
          9.400000e-01,
          5.600000e-01,
          9.000000e-01,
          1.240000e00,
          6.800000e-01,
          6.800000e-01,
          1.030000e00,
          1.030000e00,
          1.130000e00,
          1.350000e00,
          1.220000e00,
          1.110000e00,
          1.220000e00,
          8.200000e-01,
          8.600000e-01,
          7.300000e-01,
          7.900000e-01,
          1.130000e00,
          5.900000e-01,
          9.300000e-01,
          9.300000e-01,
          9.900000e-01,
          9.300000e-01,
          8.100000e-01,
          9.300000e-01,
          1.390000e00,
          1.440000e00,
          1.000000e00,
          1.120000e00,
          6.200000e-01,
          8.200000e-01,
          9.100000e-01,
          1.370000e00,
          7.500000e-01,
          9.800000e-01,
          1.490000e00,
          1.020000e00,
          1.110000e00,
          6.200000e-01,
          1.330000e00,
          1.100000e00,
          1.050000e00,
          8.400000e-01,
          8.000000e-01,
          9.200000e-01,
          1.180000e00,
          1.380000e00,
          1.010000e00,
          1.170000e00,
          1.090000e00,
          1.120000e00,
          1.170000e00,
      ],
      dtype=np.float32,
  ).reshape(8, 8)
  v4 = _assign_layout_15335824159471298539(v2)
  v5 = _assign_layout_15335824159471298539(v3)
  v6 = v5[3 : 3 + 1, 0 : 0 + 5]
  v7 = v5[3 : 3 + 1, 5 : 5 + 3]
  v8 = np.zeros(
      (
          1,
          8,
      ),
      dtype=np.float32,
  )
  v9 = v8.copy()
  v9[0 : 0 + 1, 3 : 3 + 5] = v6
  v10 = v9.copy()
  v10[0 : 0 + 1, 0 : 0 + 3] = v7
  v11 = v5[4 : 4 + 1, 0 : 0 + 5]
  v12 = v5[4 : 4 + 1, 5 : 5 + 3]
  v13 = v8.copy()
  v13[0 : 0 + 1, 3 : 3 + 5] = v11
  v14 = v13.copy()
  v14[0 : 0 + 1, 0 : 0 + 3] = v12
  v15 = v5[5 : 5 + 1, 0 : 0 + 5]
  v16 = v5[5 : 5 + 1, 5 : 5 + 3]
  v17 = v8.copy()
  v17[0 : 0 + 1, 3 : 3 + 5] = v15
  v18 = v17.copy()
  v18[0 : 0 + 1, 0 : 0 + 3] = v16
  v19 = v5[6 : 6 + 1, 0 : 0 + 2]
  v20 = v5[6 : 6 + 1, 2 : 2 + 6]
  v21 = v8.copy()
  v21[0 : 0 + 1, 6 : 6 + 2] = v19
  v22 = v21.copy()
  v22[0 : 0 + 1, 0 : 0 + 6] = v20
  v23 = v5[7 : 7 + 1, 0 : 0 + 2]
  v24 = v5[7 : 7 + 1, 2 : 2 + 6]
  v25 = v8.copy()
  v25[0 : 0 + 1, 6 : 6 + 2] = v23
  v26 = v25.copy()
  v26[0 : 0 + 1, 0 : 0 + 6] = v24
  v27 = v4[3 : 3 + 1, 0 : 0 + 5]
  v28 = v4[3 : 3 + 1, 5 : 5 + 3]
  v29 = v8.copy()
  v29[0 : 0 + 1, 3 : 3 + 5] = v27
  v30 = v29.copy()
  v30[0 : 0 + 1, 0 : 0 + 3] = v28
  v31 = v4[4 : 4 + 1, 0 : 0 + 5]
  v32 = v4[4 : 4 + 1, 5 : 5 + 3]
  v33 = v8.copy()
  v33[0 : 0 + 1, 3 : 3 + 5] = v31
  v34 = v33.copy()
  v34[0 : 0 + 1, 0 : 0 + 3] = v32
  v35 = v4[5 : 5 + 1, 0 : 0 + 5]
  v36 = v4[5 : 5 + 1, 5 : 5 + 3]
  v37 = v8.copy()
  v37[0 : 0 + 1, 3 : 3 + 5] = v35
  v38 = v37.copy()
  v38[0 : 0 + 1, 0 : 0 + 3] = v36
  v39 = v4[6 : 6 + 1, 0 : 0 + 2]
  v40 = v4[6 : 6 + 1, 2 : 2 + 6]
  v41 = v8.copy()
  v41[0 : 0 + 1, 6 : 6 + 2] = v39
  v42 = v41.copy()
  v42[0 : 0 + 1, 0 : 0 + 6] = v40
  v43 = v4[7 : 7 + 1, 0 : 0 + 2]
  v44 = v4[7 : 7 + 1, 2 : 2 + 6]
  v45 = v8.copy()
  v45[0 : 0 + 1, 6 : 6 + 2] = v43
  v46 = v45.copy()
  v46[0 : 0 + 1, 0 : 0 + 6] = v44
  v47 = v4[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v47)
  v48 = v4[1 : 1 + 1, 0 : 0 + 8].reshape(8)
  pt1 = v0.encode(v48)
  v49 = v4[2 : 2 + 1, 0 : 0 + 8].reshape(8)
  pt2 = v0.encode(v49)
  v50 = v30[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt3 = v0.encode(v50)
  v51 = v34[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt4 = v0.encode(v51)
  v52 = v38[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt5 = v0.encode(v52)
  v53 = v42[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt6 = v0.encode(v53)
  v54 = v46[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt7 = v0.encode(v54)
  v55 = v5[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt8 = v0.encode(v55)
  v56 = v5[1 : 1 + 1, 0 : 0 + 8].reshape(8)
  pt9 = v0.encode(v56)
  v57 = v5[2 : 2 + 1, 0 : 0 + 8].reshape(8)
  pt10 = v0.encode(v57)
  v58 = v10[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt11 = v0.encode(v58)
  v59 = v14[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt12 = v0.encode(v59)
  v60 = v18[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt13 = v0.encode(v60)
  v61 = v22[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt14 = v0.encode(v61)
  v62 = v26[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt15 = v0.encode(v62)
  v63 = [pt]
  v64 = [pt1]
  v65 = [pt2]
  v66 = [pt3]
  v67 = [pt4]
  v68 = [pt5]
  v69 = [pt6]
  v70 = [pt7]
  v71 = [pt8, pt9]
  v72 = [pt10, pt11]
  v73 = [pt12, pt13]
  v74 = [pt14, pt15]
  return (v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74)


def matvec_chain__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
    v4: np.ndarray,
    v5: np.ndarray,
    v6: np.ndarray,
    v7: np.ndarray,
    v8: np.ndarray,
    v9: np.ndarray,
    v10: np.ndarray,
    v11: np.ndarray,
    v12: np.ndarray,
    v13: np.ndarray,
    v14: np.ndarray,
) -> np.ndarray:
  v15 = 1
  v16 = 2
  v17 = 3
  v18 = 6
  v19 = 0
  pt = v3[0]
  pt1 = v4[0]
  pt2 = v5[0]
  pt3 = v6[0]
  pt4 = v7[0]
  pt5 = v8[0]
  pt6 = v9[0]
  pt7 = v10[0]
  pt8 = v11[0]
  pt9 = v11[1]
  pt10 = v12[0]
  pt11 = v12[1]
  pt12 = v13[0]
  pt13 = v13[1]
  pt14 = v14[0]
  pt15 = v14[1]
  ct = v2[0]
  _ct1_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct1_arg_m_in = _ct1_arg_data.shape[-1]
  _ct1_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct1_arg_m_in
  )
  _ct1_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct1_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct1_arg_r)
  )
  _ct1_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct1_arg_moduli, (int, np.integer)):
    _ct1_arg_moduli = [int(_ct1_arg_moduli)]
  ct1_arg = Polynomial(
      {
          "batch": _ct1_arg_data.shape[0],
          "num_elements": _ct1_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct1_arg_m,
          "precision": 32,
          "degree_layout": (_ct1_arg_r, _ct1_arg_c),
      },
      {"moduli": list(_ct1_arg_moduli)[:_ct1_arg_m]},
  )
  ct1_arg.polynomial = _ct1_arg_data.reshape(
      _ct1_arg_data.shape[0],
      _ct1_arg_data.shape[1],
      _ct1_arg_r,
      _ct1_arg_c,
      _ct1_arg_m_in,
  )[..., :_ct1_arg_m].copy()
  ct1_arg.batch = ct1_arg.polynomial.shape[0]
  ct1_arg.num_elements = ct1_arg.polynomial.shape[1]
  ct1_arg.num_moduli = _ct1_arg_m
  ct1_arg.degree_layout = (_ct1_arg_r, _ct1_arg_c)
  ct1_arg.r = _ct1_arg_r
  ct1_arg.c = _ct1_arg_c
  ct1_arg.moduli = list(_ct1_arg_moduli)[:_ct1_arg_m]
  ct1_arg.moduli_array = jnp.array(
      ct1_arg.moduli, dtype=getattr(ct1_arg, "modulus_dtype", jnp.uint32)
  )
  ct1_pt_ntt = (
      pt.polynomial[0, 0, :, : ct1_arg.polynomial.shape[-1]]
      .reshape(ct1_arg.r, ct1_arg.c, ct1_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct1_ptct = v0.ptct_mul[v0.max_level]
  ct1_ptct.set_plaintext(ct1_pt_ntt)
  ct1_raw = ct1_ptct.mul(ct1_arg, use_bat=False)
  _ct1_data = ct1_raw.polynomial if hasattr(ct1_raw, "polynomial") else ct1_raw
  _ct1_m_in = _ct1_data.shape[-1]
  _ct1_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct1_m_in
  )
  _ct1_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct1_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct1_r)
  )
  _ct1_moduli = getattr(ct1_raw, "moduli", v0.q_towers)
  if isinstance(_ct1_moduli, (int, np.integer)):
    _ct1_moduli = [int(_ct1_moduli)]
  ct1 = Polynomial(
      {
          "batch": _ct1_data.shape[0],
          "num_elements": _ct1_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct1_m,
          "precision": 32,
          "degree_layout": (_ct1_r, _ct1_c),
      },
      {"moduli": list(_ct1_moduli)[:_ct1_m]},
  )
  ct1.polynomial = _ct1_data.reshape(
      _ct1_data.shape[0], _ct1_data.shape[1], _ct1_r, _ct1_c, _ct1_m_in
  )[..., :_ct1_m].copy()
  ct1.batch = ct1.polynomial.shape[0]
  ct1.num_elements = ct1.polynomial.shape[1]
  ct1.num_moduli = _ct1_m
  ct1.degree_layout = (_ct1_r, _ct1_c)
  ct1.r = _ct1_r
  ct1.c = _ct1_c
  ct1.moduli = list(_ct1_moduli)[:_ct1_m]
  ct1.moduli_array = jnp.array(
      ct1.moduli, dtype=getattr(ct1, "modulus_dtype", jnp.uint32)
  )
  _ct2_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct2_arg_m_in = _ct2_arg_data.shape[-1]
  _ct2_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct2_arg_m_in
  )
  _ct2_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct2_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct2_arg_r)
  )
  _ct2_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct2_arg_moduli, (int, np.integer)):
    _ct2_arg_moduli = [int(_ct2_arg_moduli)]
  ct2_arg = Polynomial(
      {
          "batch": _ct2_arg_data.shape[0],
          "num_elements": _ct2_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct2_arg_m,
          "precision": 32,
          "degree_layout": (_ct2_arg_r, _ct2_arg_c),
      },
      {"moduli": list(_ct2_arg_moduli)[:_ct2_arg_m]},
  )
  ct2_arg.polynomial = _ct2_arg_data.reshape(
      _ct2_arg_data.shape[0],
      _ct2_arg_data.shape[1],
      _ct2_arg_r,
      _ct2_arg_c,
      _ct2_arg_m_in,
  )[..., :_ct2_arg_m].copy()
  ct2_arg.batch = ct2_arg.polynomial.shape[0]
  ct2_arg.num_elements = ct2_arg.polynomial.shape[1]
  ct2_arg.num_moduli = _ct2_arg_m
  ct2_arg.degree_layout = (_ct2_arg_r, _ct2_arg_c)
  ct2_arg.r = _ct2_arg_r
  ct2_arg.c = _ct2_arg_c
  ct2_arg.moduli = list(_ct2_arg_moduli)[:_ct2_arg_m]
  ct2_arg.moduli_array = jnp.array(
      ct2_arg.moduli, dtype=getattr(ct2_arg, "modulus_dtype", jnp.uint32)
  )
  ct2_raw = v0.he_rot[v0.max_level, 1].rotate(ct2_arg)
  _ct2_data = ct2_raw.polynomial if hasattr(ct2_raw, "polynomial") else ct2_raw
  _ct2_m_in = _ct2_data.shape[-1]
  _ct2_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct2_m_in
  )
  _ct2_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct2_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct2_r)
  )
  _ct2_moduli = getattr(ct2_raw, "moduli", v0.q_towers)
  if isinstance(_ct2_moduli, (int, np.integer)):
    _ct2_moduli = [int(_ct2_moduli)]
  ct2 = Polynomial(
      {
          "batch": _ct2_data.shape[0],
          "num_elements": _ct2_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct2_m,
          "precision": 32,
          "degree_layout": (_ct2_r, _ct2_c),
      },
      {"moduli": list(_ct2_moduli)[:_ct2_m]},
  )
  ct2.polynomial = _ct2_data.reshape(
      _ct2_data.shape[0], _ct2_data.shape[1], _ct2_r, _ct2_c, _ct2_m_in
  )[..., :_ct2_m].copy()
  ct2.batch = ct2.polynomial.shape[0]
  ct2.num_elements = ct2.polynomial.shape[1]
  ct2.num_moduli = _ct2_m
  ct2.degree_layout = (_ct2_r, _ct2_c)
  ct2.r = _ct2_r
  ct2.c = _ct2_c
  ct2.moduli = list(_ct2_moduli)[:_ct2_m]
  ct2.moduli_array = jnp.array(
      ct2.moduli, dtype=getattr(ct2, "modulus_dtype", jnp.uint32)
  )
  _ct3_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct3_arg_m_in = _ct3_arg_data.shape[-1]
  _ct3_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct3_arg_m_in
  )
  _ct3_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct3_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct3_arg_r)
  )
  _ct3_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct3_arg_moduli, (int, np.integer)):
    _ct3_arg_moduli = [int(_ct3_arg_moduli)]
  ct3_arg = Polynomial(
      {
          "batch": _ct3_arg_data.shape[0],
          "num_elements": _ct3_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct3_arg_m,
          "precision": 32,
          "degree_layout": (_ct3_arg_r, _ct3_arg_c),
      },
      {"moduli": list(_ct3_arg_moduli)[:_ct3_arg_m]},
  )
  ct3_arg.polynomial = _ct3_arg_data.reshape(
      _ct3_arg_data.shape[0],
      _ct3_arg_data.shape[1],
      _ct3_arg_r,
      _ct3_arg_c,
      _ct3_arg_m_in,
  )[..., :_ct3_arg_m].copy()
  ct3_arg.batch = ct3_arg.polynomial.shape[0]
  ct3_arg.num_elements = ct3_arg.polynomial.shape[1]
  ct3_arg.num_moduli = _ct3_arg_m
  ct3_arg.degree_layout = (_ct3_arg_r, _ct3_arg_c)
  ct3_arg.r = _ct3_arg_r
  ct3_arg.c = _ct3_arg_c
  ct3_arg.moduli = list(_ct3_arg_moduli)[:_ct3_arg_m]
  ct3_arg.moduli_array = jnp.array(
      ct3_arg.moduli, dtype=getattr(ct3_arg, "modulus_dtype", jnp.uint32)
  )
  ct3_pt_ntt = (
      pt1.polynomial[0, 0, :, : ct3_arg.polynomial.shape[-1]]
      .reshape(ct3_arg.r, ct3_arg.c, ct3_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct3_ptct = v0.ptct_mul[v0.max_level]
  ct3_ptct.set_plaintext(ct3_pt_ntt)
  ct3_raw = ct3_ptct.mul(ct3_arg, use_bat=False)
  _ct3_data = ct3_raw.polynomial if hasattr(ct3_raw, "polynomial") else ct3_raw
  _ct3_m_in = _ct3_data.shape[-1]
  _ct3_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct3_m_in
  )
  _ct3_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct3_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct3_r)
  )
  _ct3_moduli = getattr(ct3_raw, "moduli", v0.q_towers)
  if isinstance(_ct3_moduli, (int, np.integer)):
    _ct3_moduli = [int(_ct3_moduli)]
  ct3 = Polynomial(
      {
          "batch": _ct3_data.shape[0],
          "num_elements": _ct3_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct3_m,
          "precision": 32,
          "degree_layout": (_ct3_r, _ct3_c),
      },
      {"moduli": list(_ct3_moduli)[:_ct3_m]},
  )
  ct3.polynomial = _ct3_data.reshape(
      _ct3_data.shape[0], _ct3_data.shape[1], _ct3_r, _ct3_c, _ct3_m_in
  )[..., :_ct3_m].copy()
  ct3.batch = ct3.polynomial.shape[0]
  ct3.num_elements = ct3.polynomial.shape[1]
  ct3.num_moduli = _ct3_m
  ct3.degree_layout = (_ct3_r, _ct3_c)
  ct3.r = _ct3_r
  ct3.c = _ct3_c
  ct3.moduli = list(_ct3_moduli)[:_ct3_m]
  ct3.moduli_array = jnp.array(
      ct3.moduli, dtype=getattr(ct3, "modulus_dtype", jnp.uint32)
  )
  _ct4_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct4_arg_m_in = _ct4_arg_data.shape[-1]
  _ct4_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct4_arg_m_in
  )
  _ct4_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct4_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct4_arg_r)
  )
  _ct4_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct4_arg_moduli, (int, np.integer)):
    _ct4_arg_moduli = [int(_ct4_arg_moduli)]
  ct4_arg = Polynomial(
      {
          "batch": _ct4_arg_data.shape[0],
          "num_elements": _ct4_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct4_arg_m,
          "precision": 32,
          "degree_layout": (_ct4_arg_r, _ct4_arg_c),
      },
      {"moduli": list(_ct4_arg_moduli)[:_ct4_arg_m]},
  )
  ct4_arg.polynomial = _ct4_arg_data.reshape(
      _ct4_arg_data.shape[0],
      _ct4_arg_data.shape[1],
      _ct4_arg_r,
      _ct4_arg_c,
      _ct4_arg_m_in,
  )[..., :_ct4_arg_m].copy()
  ct4_arg.batch = ct4_arg.polynomial.shape[0]
  ct4_arg.num_elements = ct4_arg.polynomial.shape[1]
  ct4_arg.num_moduli = _ct4_arg_m
  ct4_arg.degree_layout = (_ct4_arg_r, _ct4_arg_c)
  ct4_arg.r = _ct4_arg_r
  ct4_arg.c = _ct4_arg_c
  ct4_arg.moduli = list(_ct4_arg_moduli)[:_ct4_arg_m]
  ct4_arg.moduli_array = jnp.array(
      ct4_arg.moduli, dtype=getattr(ct4_arg, "modulus_dtype", jnp.uint32)
  )
  ct4_raw = v0.he_rot[v0.max_level, 2].rotate(ct4_arg)
  _ct4_data = ct4_raw.polynomial if hasattr(ct4_raw, "polynomial") else ct4_raw
  _ct4_m_in = _ct4_data.shape[-1]
  _ct4_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct4_m_in
  )
  _ct4_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct4_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct4_r)
  )
  _ct4_moduli = getattr(ct4_raw, "moduli", v0.q_towers)
  if isinstance(_ct4_moduli, (int, np.integer)):
    _ct4_moduli = [int(_ct4_moduli)]
  ct4 = Polynomial(
      {
          "batch": _ct4_data.shape[0],
          "num_elements": _ct4_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct4_m,
          "precision": 32,
          "degree_layout": (_ct4_r, _ct4_c),
      },
      {"moduli": list(_ct4_moduli)[:_ct4_m]},
  )
  ct4.polynomial = _ct4_data.reshape(
      _ct4_data.shape[0], _ct4_data.shape[1], _ct4_r, _ct4_c, _ct4_m_in
  )[..., :_ct4_m].copy()
  ct4.batch = ct4.polynomial.shape[0]
  ct4.num_elements = ct4.polynomial.shape[1]
  ct4.num_moduli = _ct4_m
  ct4.degree_layout = (_ct4_r, _ct4_c)
  ct4.r = _ct4_r
  ct4.c = _ct4_c
  ct4.moduli = list(_ct4_moduli)[:_ct4_m]
  ct4.moduli_array = jnp.array(
      ct4.moduli, dtype=getattr(ct4, "modulus_dtype", jnp.uint32)
  )
  _ct5_arg_data = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  _ct5_arg_m_in = _ct5_arg_data.shape[-1]
  _ct5_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct5_arg_m_in
  )
  _ct5_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct5_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct5_arg_r)
  )
  _ct5_arg_moduli = getattr(ct4, "moduli", v0.q_towers)
  if isinstance(_ct5_arg_moduli, (int, np.integer)):
    _ct5_arg_moduli = [int(_ct5_arg_moduli)]
  ct5_arg = Polynomial(
      {
          "batch": _ct5_arg_data.shape[0],
          "num_elements": _ct5_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct5_arg_m,
          "precision": 32,
          "degree_layout": (_ct5_arg_r, _ct5_arg_c),
      },
      {"moduli": list(_ct5_arg_moduli)[:_ct5_arg_m]},
  )
  ct5_arg.polynomial = _ct5_arg_data.reshape(
      _ct5_arg_data.shape[0],
      _ct5_arg_data.shape[1],
      _ct5_arg_r,
      _ct5_arg_c,
      _ct5_arg_m_in,
  )[..., :_ct5_arg_m].copy()
  ct5_arg.batch = ct5_arg.polynomial.shape[0]
  ct5_arg.num_elements = ct5_arg.polynomial.shape[1]
  ct5_arg.num_moduli = _ct5_arg_m
  ct5_arg.degree_layout = (_ct5_arg_r, _ct5_arg_c)
  ct5_arg.r = _ct5_arg_r
  ct5_arg.c = _ct5_arg_c
  ct5_arg.moduli = list(_ct5_arg_moduli)[:_ct5_arg_m]
  ct5_arg.moduli_array = jnp.array(
      ct5_arg.moduli, dtype=getattr(ct5_arg, "modulus_dtype", jnp.uint32)
  )
  ct5_pt_ntt = (
      pt2.polynomial[0, 0, :, : ct5_arg.polynomial.shape[-1]]
      .reshape(ct5_arg.r, ct5_arg.c, ct5_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5_raw = ct5_ptct.mul(ct5_arg, use_bat=False)
  _ct5_data = ct5_raw.polynomial if hasattr(ct5_raw, "polynomial") else ct5_raw
  _ct5_m_in = _ct5_data.shape[-1]
  _ct5_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct5_m_in
  )
  _ct5_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct5_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct5_r)
  )
  _ct5_moduli = getattr(ct5_raw, "moduli", v0.q_towers)
  if isinstance(_ct5_moduli, (int, np.integer)):
    _ct5_moduli = [int(_ct5_moduli)]
  ct5 = Polynomial(
      {
          "batch": _ct5_data.shape[0],
          "num_elements": _ct5_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct5_m,
          "precision": 32,
          "degree_layout": (_ct5_r, _ct5_c),
      },
      {"moduli": list(_ct5_moduli)[:_ct5_m]},
  )
  ct5.polynomial = _ct5_data.reshape(
      _ct5_data.shape[0], _ct5_data.shape[1], _ct5_r, _ct5_c, _ct5_m_in
  )[..., :_ct5_m].copy()
  ct5.batch = ct5.polynomial.shape[0]
  ct5.num_elements = ct5.polynomial.shape[1]
  ct5.num_moduli = _ct5_m
  ct5.degree_layout = (_ct5_r, _ct5_c)
  ct5.r = _ct5_r
  ct5.c = _ct5_c
  ct5.moduli = list(_ct5_moduli)[:_ct5_m]
  ct5.moduli_array = jnp.array(
      ct5.moduli, dtype=getattr(ct5, "modulus_dtype", jnp.uint32)
  )
  _ct6_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct6_arg_m_in = _ct6_arg_data.shape[-1]
  _ct6_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct6_arg_m_in
  )
  _ct6_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct6_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct6_arg_r)
  )
  _ct6_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct6_arg_moduli, (int, np.integer)):
    _ct6_arg_moduli = [int(_ct6_arg_moduli)]
  ct6_arg = Polynomial(
      {
          "batch": _ct6_arg_data.shape[0],
          "num_elements": _ct6_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct6_arg_m,
          "precision": 32,
          "degree_layout": (_ct6_arg_r, _ct6_arg_c),
      },
      {"moduli": list(_ct6_arg_moduli)[:_ct6_arg_m]},
  )
  ct6_arg.polynomial = _ct6_arg_data.reshape(
      _ct6_arg_data.shape[0],
      _ct6_arg_data.shape[1],
      _ct6_arg_r,
      _ct6_arg_c,
      _ct6_arg_m_in,
  )[..., :_ct6_arg_m].copy()
  ct6_arg.batch = ct6_arg.polynomial.shape[0]
  ct6_arg.num_elements = ct6_arg.polynomial.shape[1]
  ct6_arg.num_moduli = _ct6_arg_m
  ct6_arg.degree_layout = (_ct6_arg_r, _ct6_arg_c)
  ct6_arg.r = _ct6_arg_r
  ct6_arg.c = _ct6_arg_c
  ct6_arg.moduli = list(_ct6_arg_moduli)[:_ct6_arg_m]
  ct6_arg.moduli_array = jnp.array(
      ct6_arg.moduli, dtype=getattr(ct6_arg, "modulus_dtype", jnp.uint32)
  )
  ct6_pt_ntt = (
      pt3.polynomial[0, 0, :, : ct6_arg.polynomial.shape[-1]]
      .reshape(ct6_arg.r, ct6_arg.c, ct6_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct6_ptct = v0.ptct_mul[v0.max_level]
  ct6_ptct.set_plaintext(ct6_pt_ntt)
  ct6_raw = ct6_ptct.mul(ct6_arg, use_bat=False)
  _ct6_data = ct6_raw.polynomial if hasattr(ct6_raw, "polynomial") else ct6_raw
  _ct6_m_in = _ct6_data.shape[-1]
  _ct6_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct6_m_in
  )
  _ct6_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct6_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct6_r)
  )
  _ct6_moduli = getattr(ct6_raw, "moduli", v0.q_towers)
  if isinstance(_ct6_moduli, (int, np.integer)):
    _ct6_moduli = [int(_ct6_moduli)]
  ct6 = Polynomial(
      {
          "batch": _ct6_data.shape[0],
          "num_elements": _ct6_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct6_m,
          "precision": 32,
          "degree_layout": (_ct6_r, _ct6_c),
      },
      {"moduli": list(_ct6_moduli)[:_ct6_m]},
  )
  ct6.polynomial = _ct6_data.reshape(
      _ct6_data.shape[0], _ct6_data.shape[1], _ct6_r, _ct6_c, _ct6_m_in
  )[..., :_ct6_m].copy()
  ct6.batch = ct6.polynomial.shape[0]
  ct6.num_elements = ct6.polynomial.shape[1]
  ct6.num_moduli = _ct6_m
  ct6.degree_layout = (_ct6_r, _ct6_c)
  ct6.r = _ct6_r
  ct6.c = _ct6_c
  ct6.moduli = list(_ct6_moduli)[:_ct6_m]
  ct6.moduli_array = jnp.array(
      ct6.moduli, dtype=getattr(ct6, "modulus_dtype", jnp.uint32)
  )
  _ct7_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct7_arg_m_in = _ct7_arg_data.shape[-1]
  _ct7_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct7_arg_m_in
  )
  _ct7_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct7_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct7_arg_r)
  )
  _ct7_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct7_arg_moduli, (int, np.integer)):
    _ct7_arg_moduli = [int(_ct7_arg_moduli)]
  ct7_arg = Polynomial(
      {
          "batch": _ct7_arg_data.shape[0],
          "num_elements": _ct7_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct7_arg_m,
          "precision": 32,
          "degree_layout": (_ct7_arg_r, _ct7_arg_c),
      },
      {"moduli": list(_ct7_arg_moduli)[:_ct7_arg_m]},
  )
  ct7_arg.polynomial = _ct7_arg_data.reshape(
      _ct7_arg_data.shape[0],
      _ct7_arg_data.shape[1],
      _ct7_arg_r,
      _ct7_arg_c,
      _ct7_arg_m_in,
  )[..., :_ct7_arg_m].copy()
  ct7_arg.batch = ct7_arg.polynomial.shape[0]
  ct7_arg.num_elements = ct7_arg.polynomial.shape[1]
  ct7_arg.num_moduli = _ct7_arg_m
  ct7_arg.degree_layout = (_ct7_arg_r, _ct7_arg_c)
  ct7_arg.r = _ct7_arg_r
  ct7_arg.c = _ct7_arg_c
  ct7_arg.moduli = list(_ct7_arg_moduli)[:_ct7_arg_m]
  ct7_arg.moduli_array = jnp.array(
      ct7_arg.moduli, dtype=getattr(ct7_arg, "modulus_dtype", jnp.uint32)
  )
  ct7_pt_ntt = (
      pt4.polynomial[0, 0, :, : ct7_arg.polynomial.shape[-1]]
      .reshape(ct7_arg.r, ct7_arg.c, ct7_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct7_ptct = v0.ptct_mul[v0.max_level]
  ct7_ptct.set_plaintext(ct7_pt_ntt)
  ct7_raw = ct7_ptct.mul(ct7_arg, use_bat=False)
  _ct7_data = ct7_raw.polynomial if hasattr(ct7_raw, "polynomial") else ct7_raw
  _ct7_m_in = _ct7_data.shape[-1]
  _ct7_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct7_m_in
  )
  _ct7_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct7_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct7_r)
  )
  _ct7_moduli = getattr(ct7_raw, "moduli", v0.q_towers)
  if isinstance(_ct7_moduli, (int, np.integer)):
    _ct7_moduli = [int(_ct7_moduli)]
  ct7 = Polynomial(
      {
          "batch": _ct7_data.shape[0],
          "num_elements": _ct7_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct7_m,
          "precision": 32,
          "degree_layout": (_ct7_r, _ct7_c),
      },
      {"moduli": list(_ct7_moduli)[:_ct7_m]},
  )
  ct7.polynomial = _ct7_data.reshape(
      _ct7_data.shape[0], _ct7_data.shape[1], _ct7_r, _ct7_c, _ct7_m_in
  )[..., :_ct7_m].copy()
  ct7.batch = ct7.polynomial.shape[0]
  ct7.num_elements = ct7.polynomial.shape[1]
  ct7.num_moduli = _ct7_m
  ct7.degree_layout = (_ct7_r, _ct7_c)
  ct7.r = _ct7_r
  ct7.c = _ct7_c
  ct7.moduli = list(_ct7_moduli)[:_ct7_m]
  ct7.moduli_array = jnp.array(
      ct7.moduli, dtype=getattr(ct7, "modulus_dtype", jnp.uint32)
  )
  _ct8_arg_data = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  _ct8_arg_m_in = _ct8_arg_data.shape[-1]
  _ct8_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct8_arg_m_in
  )
  _ct8_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct8_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct8_arg_r)
  )
  _ct8_arg_moduli = getattr(ct4, "moduli", v0.q_towers)
  if isinstance(_ct8_arg_moduli, (int, np.integer)):
    _ct8_arg_moduli = [int(_ct8_arg_moduli)]
  ct8_arg = Polynomial(
      {
          "batch": _ct8_arg_data.shape[0],
          "num_elements": _ct8_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct8_arg_m,
          "precision": 32,
          "degree_layout": (_ct8_arg_r, _ct8_arg_c),
      },
      {"moduli": list(_ct8_arg_moduli)[:_ct8_arg_m]},
  )
  ct8_arg.polynomial = _ct8_arg_data.reshape(
      _ct8_arg_data.shape[0],
      _ct8_arg_data.shape[1],
      _ct8_arg_r,
      _ct8_arg_c,
      _ct8_arg_m_in,
  )[..., :_ct8_arg_m].copy()
  ct8_arg.batch = ct8_arg.polynomial.shape[0]
  ct8_arg.num_elements = ct8_arg.polynomial.shape[1]
  ct8_arg.num_moduli = _ct8_arg_m
  ct8_arg.degree_layout = (_ct8_arg_r, _ct8_arg_c)
  ct8_arg.r = _ct8_arg_r
  ct8_arg.c = _ct8_arg_c
  ct8_arg.moduli = list(_ct8_arg_moduli)[:_ct8_arg_m]
  ct8_arg.moduli_array = jnp.array(
      ct8_arg.moduli, dtype=getattr(ct8_arg, "modulus_dtype", jnp.uint32)
  )
  ct8_pt_ntt = (
      pt5.polynomial[0, 0, :, : ct8_arg.polynomial.shape[-1]]
      .reshape(ct8_arg.r, ct8_arg.c, ct8_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct8_ptct = v0.ptct_mul[v0.max_level]
  ct8_ptct.set_plaintext(ct8_pt_ntt)
  ct8_raw = ct8_ptct.mul(ct8_arg, use_bat=False)
  _ct8_data = ct8_raw.polynomial if hasattr(ct8_raw, "polynomial") else ct8_raw
  _ct8_m_in = _ct8_data.shape[-1]
  _ct8_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct8_m_in
  )
  _ct8_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct8_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct8_r)
  )
  _ct8_moduli = getattr(ct8_raw, "moduli", v0.q_towers)
  if isinstance(_ct8_moduli, (int, np.integer)):
    _ct8_moduli = [int(_ct8_moduli)]
  ct8 = Polynomial(
      {
          "batch": _ct8_data.shape[0],
          "num_elements": _ct8_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct8_m,
          "precision": 32,
          "degree_layout": (_ct8_r, _ct8_c),
      },
      {"moduli": list(_ct8_moduli)[:_ct8_m]},
  )
  ct8.polynomial = _ct8_data.reshape(
      _ct8_data.shape[0], _ct8_data.shape[1], _ct8_r, _ct8_c, _ct8_m_in
  )[..., :_ct8_m].copy()
  ct8.batch = ct8.polynomial.shape[0]
  ct8.num_elements = ct8.polynomial.shape[1]
  ct8.num_moduli = _ct8_m
  ct8.degree_layout = (_ct8_r, _ct8_c)
  ct8.r = _ct8_r
  ct8.c = _ct8_c
  ct8.moduli = list(_ct8_moduli)[:_ct8_m]
  ct8.moduli_array = jnp.array(
      ct8.moduli, dtype=getattr(ct8, "modulus_dtype", jnp.uint32)
  )
  _ct9_data = ct6.polynomial if hasattr(ct6, "polynomial") else ct6
  _ct9_m_in = _ct9_data.shape[-1]
  _ct9_m = _ct9_m_in
  _ct9_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct9_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct9_r)
  )
  _ct9_moduli = getattr(ct6, "moduli", v0.q_towers)
  if isinstance(_ct9_moduli, (int, np.integer)):
    _ct9_moduli = [int(_ct9_moduli)]
  ct9 = Polynomial(
      {
          "batch": _ct9_data.shape[0],
          "num_elements": _ct9_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct9_m,
          "precision": 32,
          "degree_layout": (_ct9_r, _ct9_c),
      },
      {"moduli": list(_ct9_moduli)[:_ct9_m]},
  )
  ct9.polynomial = _ct9_data.reshape(
      _ct9_data.shape[0], _ct9_data.shape[1], _ct9_r, _ct9_c, _ct9_m_in
  )[..., :_ct9_m].copy()
  ct9.batch = ct9.polynomial.shape[0]
  ct9.num_elements = ct9.polynomial.shape[1]
  ct9.num_moduli = _ct9_m
  ct9.degree_layout = (_ct9_r, _ct9_c)
  ct9.r = _ct9_r
  ct9.c = _ct9_c
  ct9.moduli = list(_ct9_moduli)[:_ct9_m]
  ct9.moduli_array = jnp.array(
      ct9.moduli, dtype=getattr(ct9, "modulus_dtype", jnp.uint32)
  )
  _ct9_rhs_data = ct7.polynomial if hasattr(ct7, "polynomial") else ct7
  _ct9_rhs_m_in = _ct9_rhs_data.shape[-1]
  _ct9_rhs_m = _ct9_rhs_m_in
  _ct9_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct9_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct9_rhs_r)
  )
  _ct9_rhs_moduli = getattr(ct7, "moduli", v0.q_towers)
  if isinstance(_ct9_rhs_moduli, (int, np.integer)):
    _ct9_rhs_moduli = [int(_ct9_rhs_moduli)]
  ct9_rhs = Polynomial(
      {
          "batch": _ct9_rhs_data.shape[0],
          "num_elements": _ct9_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct9_rhs_m,
          "precision": 32,
          "degree_layout": (_ct9_rhs_r, _ct9_rhs_c),
      },
      {"moduli": list(_ct9_rhs_moduli)[:_ct9_rhs_m]},
  )
  ct9_rhs.polynomial = _ct9_rhs_data.reshape(
      _ct9_rhs_data.shape[0],
      _ct9_rhs_data.shape[1],
      _ct9_rhs_r,
      _ct9_rhs_c,
      _ct9_rhs_m_in,
  )[..., :_ct9_rhs_m].copy()
  ct9_rhs.batch = ct9_rhs.polynomial.shape[0]
  ct9_rhs.num_elements = ct9_rhs.polynomial.shape[1]
  ct9_rhs.num_moduli = _ct9_rhs_m
  ct9_rhs.degree_layout = (_ct9_rhs_r, _ct9_rhs_c)
  ct9_rhs.r = _ct9_rhs_r
  ct9_rhs.c = _ct9_rhs_c
  ct9_rhs.moduli = list(_ct9_rhs_moduli)[:_ct9_rhs_m]
  ct9_rhs.moduli_array = jnp.array(
      ct9_rhs.moduli, dtype=getattr(ct9_rhs, "modulus_dtype", jnp.uint32)
  )
  ct9.add(ct9_rhs)
  _moduli = jnp.array(ct9.moduli, dtype=jnp.uint32)
  ct9.polynomial = jnp.where(
      ct9.polynomial >= _moduli, ct9.polynomial - _moduli, ct9.polynomial
  )
  _ct10_data = ct9.polynomial if hasattr(ct9, "polynomial") else ct9
  _ct10_m_in = _ct10_data.shape[-1]
  _ct10_m = _ct10_m_in
  _ct10_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct10_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct10_r)
  )
  _ct10_moduli = getattr(ct9, "moduli", v0.q_towers)
  if isinstance(_ct10_moduli, (int, np.integer)):
    _ct10_moduli = [int(_ct10_moduli)]
  ct10 = Polynomial(
      {
          "batch": _ct10_data.shape[0],
          "num_elements": _ct10_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct10_m,
          "precision": 32,
          "degree_layout": (_ct10_r, _ct10_c),
      },
      {"moduli": list(_ct10_moduli)[:_ct10_m]},
  )
  ct10.polynomial = _ct10_data.reshape(
      _ct10_data.shape[0], _ct10_data.shape[1], _ct10_r, _ct10_c, _ct10_m_in
  )[..., :_ct10_m].copy()
  ct10.batch = ct10.polynomial.shape[0]
  ct10.num_elements = ct10.polynomial.shape[1]
  ct10.num_moduli = _ct10_m
  ct10.degree_layout = (_ct10_r, _ct10_c)
  ct10.r = _ct10_r
  ct10.c = _ct10_c
  ct10.moduli = list(_ct10_moduli)[:_ct10_m]
  ct10.moduli_array = jnp.array(
      ct10.moduli, dtype=getattr(ct10, "modulus_dtype", jnp.uint32)
  )
  _ct10_rhs_data = ct8.polynomial if hasattr(ct8, "polynomial") else ct8
  _ct10_rhs_m_in = _ct10_rhs_data.shape[-1]
  _ct10_rhs_m = _ct10_rhs_m_in
  _ct10_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct10_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct10_rhs_r)
  )
  _ct10_rhs_moduli = getattr(ct8, "moduli", v0.q_towers)
  if isinstance(_ct10_rhs_moduli, (int, np.integer)):
    _ct10_rhs_moduli = [int(_ct10_rhs_moduli)]
  ct10_rhs = Polynomial(
      {
          "batch": _ct10_rhs_data.shape[0],
          "num_elements": _ct10_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct10_rhs_m,
          "precision": 32,
          "degree_layout": (_ct10_rhs_r, _ct10_rhs_c),
      },
      {"moduli": list(_ct10_rhs_moduli)[:_ct10_rhs_m]},
  )
  ct10_rhs.polynomial = _ct10_rhs_data.reshape(
      _ct10_rhs_data.shape[0],
      _ct10_rhs_data.shape[1],
      _ct10_rhs_r,
      _ct10_rhs_c,
      _ct10_rhs_m_in,
  )[..., :_ct10_rhs_m].copy()
  ct10_rhs.batch = ct10_rhs.polynomial.shape[0]
  ct10_rhs.num_elements = ct10_rhs.polynomial.shape[1]
  ct10_rhs.num_moduli = _ct10_rhs_m
  ct10_rhs.degree_layout = (_ct10_rhs_r, _ct10_rhs_c)
  ct10_rhs.r = _ct10_rhs_r
  ct10_rhs.c = _ct10_rhs_c
  ct10_rhs.moduli = list(_ct10_rhs_moduli)[:_ct10_rhs_m]
  ct10_rhs.moduli_array = jnp.array(
      ct10_rhs.moduli, dtype=getattr(ct10_rhs, "modulus_dtype", jnp.uint32)
  )
  ct10.add(ct10_rhs)
  _moduli = jnp.array(ct10.moduli, dtype=jnp.uint32)
  ct10.polynomial = jnp.where(
      ct10.polynomial >= _moduli, ct10.polynomial - _moduli, ct10.polynomial
  )
  _ct11_arg_data = ct10.polynomial if hasattr(ct10, "polynomial") else ct10
  _ct11_arg_m_in = _ct11_arg_data.shape[-1]
  _ct11_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct11_arg_m_in
  )
  _ct11_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct11_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct11_arg_r)
  )
  _ct11_arg_moduli = getattr(ct10, "moduli", v0.q_towers)
  if isinstance(_ct11_arg_moduli, (int, np.integer)):
    _ct11_arg_moduli = [int(_ct11_arg_moduli)]
  ct11_arg = Polynomial(
      {
          "batch": _ct11_arg_data.shape[0],
          "num_elements": _ct11_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct11_arg_m,
          "precision": 32,
          "degree_layout": (_ct11_arg_r, _ct11_arg_c),
      },
      {"moduli": list(_ct11_arg_moduli)[:_ct11_arg_m]},
  )
  ct11_arg.polynomial = _ct11_arg_data.reshape(
      _ct11_arg_data.shape[0],
      _ct11_arg_data.shape[1],
      _ct11_arg_r,
      _ct11_arg_c,
      _ct11_arg_m_in,
  )[..., :_ct11_arg_m].copy()
  ct11_arg.batch = ct11_arg.polynomial.shape[0]
  ct11_arg.num_elements = ct11_arg.polynomial.shape[1]
  ct11_arg.num_moduli = _ct11_arg_m
  ct11_arg.degree_layout = (_ct11_arg_r, _ct11_arg_c)
  ct11_arg.r = _ct11_arg_r
  ct11_arg.c = _ct11_arg_c
  ct11_arg.moduli = list(_ct11_arg_moduli)[:_ct11_arg_m]
  ct11_arg.moduli_array = jnp.array(
      ct11_arg.moduli, dtype=getattr(ct11_arg, "modulus_dtype", jnp.uint32)
  )
  ct11_raw = v0.he_rot[v0.max_level, 3].rotate(ct11_arg)
  _ct11_data = (
      ct11_raw.polynomial if hasattr(ct11_raw, "polynomial") else ct11_raw
  )
  _ct11_m_in = _ct11_data.shape[-1]
  _ct11_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct11_m_in
  )
  _ct11_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct11_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct11_r)
  )
  _ct11_moduli = getattr(ct11_raw, "moduli", v0.q_towers)
  if isinstance(_ct11_moduli, (int, np.integer)):
    _ct11_moduli = [int(_ct11_moduli)]
  ct11 = Polynomial(
      {
          "batch": _ct11_data.shape[0],
          "num_elements": _ct11_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct11_m,
          "precision": 32,
          "degree_layout": (_ct11_r, _ct11_c),
      },
      {"moduli": list(_ct11_moduli)[:_ct11_m]},
  )
  ct11.polynomial = _ct11_data.reshape(
      _ct11_data.shape[0], _ct11_data.shape[1], _ct11_r, _ct11_c, _ct11_m_in
  )[..., :_ct11_m].copy()
  ct11.batch = ct11.polynomial.shape[0]
  ct11.num_elements = ct11.polynomial.shape[1]
  ct11.num_moduli = _ct11_m
  ct11.degree_layout = (_ct11_r, _ct11_c)
  ct11.r = _ct11_r
  ct11.c = _ct11_c
  ct11.moduli = list(_ct11_moduli)[:_ct11_m]
  ct11.moduli_array = jnp.array(
      ct11.moduli, dtype=getattr(ct11, "modulus_dtype", jnp.uint32)
  )
  _ct12_arg_data = ct.polynomial if hasattr(ct, "polynomial") else ct
  _ct12_arg_m_in = _ct12_arg_data.shape[-1]
  _ct12_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct12_arg_m_in
  )
  _ct12_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct12_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct12_arg_r)
  )
  _ct12_arg_moduli = getattr(ct, "moduli", v0.q_towers)
  if isinstance(_ct12_arg_moduli, (int, np.integer)):
    _ct12_arg_moduli = [int(_ct12_arg_moduli)]
  ct12_arg = Polynomial(
      {
          "batch": _ct12_arg_data.shape[0],
          "num_elements": _ct12_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct12_arg_m,
          "precision": 32,
          "degree_layout": (_ct12_arg_r, _ct12_arg_c),
      },
      {"moduli": list(_ct12_arg_moduli)[:_ct12_arg_m]},
  )
  ct12_arg.polynomial = _ct12_arg_data.reshape(
      _ct12_arg_data.shape[0],
      _ct12_arg_data.shape[1],
      _ct12_arg_r,
      _ct12_arg_c,
      _ct12_arg_m_in,
  )[..., :_ct12_arg_m].copy()
  ct12_arg.batch = ct12_arg.polynomial.shape[0]
  ct12_arg.num_elements = ct12_arg.polynomial.shape[1]
  ct12_arg.num_moduli = _ct12_arg_m
  ct12_arg.degree_layout = (_ct12_arg_r, _ct12_arg_c)
  ct12_arg.r = _ct12_arg_r
  ct12_arg.c = _ct12_arg_c
  ct12_arg.moduli = list(_ct12_arg_moduli)[:_ct12_arg_m]
  ct12_arg.moduli_array = jnp.array(
      ct12_arg.moduli, dtype=getattr(ct12_arg, "modulus_dtype", jnp.uint32)
  )
  ct12_pt_ntt = (
      pt6.polynomial[0, 0, :, : ct12_arg.polynomial.shape[-1]]
      .reshape(ct12_arg.r, ct12_arg.c, ct12_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct12_ptct = v0.ptct_mul[v0.max_level]
  ct12_ptct.set_plaintext(ct12_pt_ntt)
  ct12_raw = ct12_ptct.mul(ct12_arg, use_bat=False)
  _ct12_data = (
      ct12_raw.polynomial if hasattr(ct12_raw, "polynomial") else ct12_raw
  )
  _ct12_m_in = _ct12_data.shape[-1]
  _ct12_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct12_m_in
  )
  _ct12_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct12_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct12_r)
  )
  _ct12_moduli = getattr(ct12_raw, "moduli", v0.q_towers)
  if isinstance(_ct12_moduli, (int, np.integer)):
    _ct12_moduli = [int(_ct12_moduli)]
  ct12 = Polynomial(
      {
          "batch": _ct12_data.shape[0],
          "num_elements": _ct12_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct12_m,
          "precision": 32,
          "degree_layout": (_ct12_r, _ct12_c),
      },
      {"moduli": list(_ct12_moduli)[:_ct12_m]},
  )
  ct12.polynomial = _ct12_data.reshape(
      _ct12_data.shape[0], _ct12_data.shape[1], _ct12_r, _ct12_c, _ct12_m_in
  )[..., :_ct12_m].copy()
  ct12.batch = ct12.polynomial.shape[0]
  ct12.num_elements = ct12.polynomial.shape[1]
  ct12.num_moduli = _ct12_m
  ct12.degree_layout = (_ct12_r, _ct12_c)
  ct12.r = _ct12_r
  ct12.c = _ct12_c
  ct12.moduli = list(_ct12_moduli)[:_ct12_m]
  ct12.moduli_array = jnp.array(
      ct12.moduli, dtype=getattr(ct12, "modulus_dtype", jnp.uint32)
  )
  _ct13_arg_data = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  _ct13_arg_m_in = _ct13_arg_data.shape[-1]
  _ct13_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct13_arg_m_in
  )
  _ct13_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct13_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct13_arg_r)
  )
  _ct13_arg_moduli = getattr(ct2, "moduli", v0.q_towers)
  if isinstance(_ct13_arg_moduli, (int, np.integer)):
    _ct13_arg_moduli = [int(_ct13_arg_moduli)]
  ct13_arg = Polynomial(
      {
          "batch": _ct13_arg_data.shape[0],
          "num_elements": _ct13_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct13_arg_m,
          "precision": 32,
          "degree_layout": (_ct13_arg_r, _ct13_arg_c),
      },
      {"moduli": list(_ct13_arg_moduli)[:_ct13_arg_m]},
  )
  ct13_arg.polynomial = _ct13_arg_data.reshape(
      _ct13_arg_data.shape[0],
      _ct13_arg_data.shape[1],
      _ct13_arg_r,
      _ct13_arg_c,
      _ct13_arg_m_in,
  )[..., :_ct13_arg_m].copy()
  ct13_arg.batch = ct13_arg.polynomial.shape[0]
  ct13_arg.num_elements = ct13_arg.polynomial.shape[1]
  ct13_arg.num_moduli = _ct13_arg_m
  ct13_arg.degree_layout = (_ct13_arg_r, _ct13_arg_c)
  ct13_arg.r = _ct13_arg_r
  ct13_arg.c = _ct13_arg_c
  ct13_arg.moduli = list(_ct13_arg_moduli)[:_ct13_arg_m]
  ct13_arg.moduli_array = jnp.array(
      ct13_arg.moduli, dtype=getattr(ct13_arg, "modulus_dtype", jnp.uint32)
  )
  ct13_pt_ntt = (
      pt7.polynomial[0, 0, :, : ct13_arg.polynomial.shape[-1]]
      .reshape(ct13_arg.r, ct13_arg.c, ct13_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct13_ptct = v0.ptct_mul[v0.max_level]
  ct13_ptct.set_plaintext(ct13_pt_ntt)
  ct13_raw = ct13_ptct.mul(ct13_arg, use_bat=False)
  _ct13_data = (
      ct13_raw.polynomial if hasattr(ct13_raw, "polynomial") else ct13_raw
  )
  _ct13_m_in = _ct13_data.shape[-1]
  _ct13_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct13_m_in
  )
  _ct13_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct13_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct13_r)
  )
  _ct13_moduli = getattr(ct13_raw, "moduli", v0.q_towers)
  if isinstance(_ct13_moduli, (int, np.integer)):
    _ct13_moduli = [int(_ct13_moduli)]
  ct13 = Polynomial(
      {
          "batch": _ct13_data.shape[0],
          "num_elements": _ct13_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct13_m,
          "precision": 32,
          "degree_layout": (_ct13_r, _ct13_c),
      },
      {"moduli": list(_ct13_moduli)[:_ct13_m]},
  )
  ct13.polynomial = _ct13_data.reshape(
      _ct13_data.shape[0], _ct13_data.shape[1], _ct13_r, _ct13_c, _ct13_m_in
  )[..., :_ct13_m].copy()
  ct13.batch = ct13.polynomial.shape[0]
  ct13.num_elements = ct13.polynomial.shape[1]
  ct13.num_moduli = _ct13_m
  ct13.degree_layout = (_ct13_r, _ct13_c)
  ct13.r = _ct13_r
  ct13.c = _ct13_c
  ct13.moduli = list(_ct13_moduli)[:_ct13_m]
  ct13.moduli_array = jnp.array(
      ct13.moduli, dtype=getattr(ct13, "modulus_dtype", jnp.uint32)
  )
  _ct14_data = ct12.polynomial if hasattr(ct12, "polynomial") else ct12
  _ct14_m_in = _ct14_data.shape[-1]
  _ct14_m = _ct14_m_in
  _ct14_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct14_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct14_r)
  )
  _ct14_moduli = getattr(ct12, "moduli", v0.q_towers)
  if isinstance(_ct14_moduli, (int, np.integer)):
    _ct14_moduli = [int(_ct14_moduli)]
  ct14 = Polynomial(
      {
          "batch": _ct14_data.shape[0],
          "num_elements": _ct14_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct14_m,
          "precision": 32,
          "degree_layout": (_ct14_r, _ct14_c),
      },
      {"moduli": list(_ct14_moduli)[:_ct14_m]},
  )
  ct14.polynomial = _ct14_data.reshape(
      _ct14_data.shape[0], _ct14_data.shape[1], _ct14_r, _ct14_c, _ct14_m_in
  )[..., :_ct14_m].copy()
  ct14.batch = ct14.polynomial.shape[0]
  ct14.num_elements = ct14.polynomial.shape[1]
  ct14.num_moduli = _ct14_m
  ct14.degree_layout = (_ct14_r, _ct14_c)
  ct14.r = _ct14_r
  ct14.c = _ct14_c
  ct14.moduli = list(_ct14_moduli)[:_ct14_m]
  ct14.moduli_array = jnp.array(
      ct14.moduli, dtype=getattr(ct14, "modulus_dtype", jnp.uint32)
  )
  _ct14_rhs_data = ct13.polynomial if hasattr(ct13, "polynomial") else ct13
  _ct14_rhs_m_in = _ct14_rhs_data.shape[-1]
  _ct14_rhs_m = _ct14_rhs_m_in
  _ct14_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct14_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct14_rhs_r)
  )
  _ct14_rhs_moduli = getattr(ct13, "moduli", v0.q_towers)
  if isinstance(_ct14_rhs_moduli, (int, np.integer)):
    _ct14_rhs_moduli = [int(_ct14_rhs_moduli)]
  ct14_rhs = Polynomial(
      {
          "batch": _ct14_rhs_data.shape[0],
          "num_elements": _ct14_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct14_rhs_m,
          "precision": 32,
          "degree_layout": (_ct14_rhs_r, _ct14_rhs_c),
      },
      {"moduli": list(_ct14_rhs_moduli)[:_ct14_rhs_m]},
  )
  ct14_rhs.polynomial = _ct14_rhs_data.reshape(
      _ct14_rhs_data.shape[0],
      _ct14_rhs_data.shape[1],
      _ct14_rhs_r,
      _ct14_rhs_c,
      _ct14_rhs_m_in,
  )[..., :_ct14_rhs_m].copy()
  ct14_rhs.batch = ct14_rhs.polynomial.shape[0]
  ct14_rhs.num_elements = ct14_rhs.polynomial.shape[1]
  ct14_rhs.num_moduli = _ct14_rhs_m
  ct14_rhs.degree_layout = (_ct14_rhs_r, _ct14_rhs_c)
  ct14_rhs.r = _ct14_rhs_r
  ct14_rhs.c = _ct14_rhs_c
  ct14_rhs.moduli = list(_ct14_rhs_moduli)[:_ct14_rhs_m]
  ct14_rhs.moduli_array = jnp.array(
      ct14_rhs.moduli, dtype=getattr(ct14_rhs, "modulus_dtype", jnp.uint32)
  )
  ct14.add(ct14_rhs)
  _moduli = jnp.array(ct14.moduli, dtype=jnp.uint32)
  ct14.polynomial = jnp.where(
      ct14.polynomial >= _moduli, ct14.polynomial - _moduli, ct14.polynomial
  )
  _ct15_arg_data = ct14.polynomial if hasattr(ct14, "polynomial") else ct14
  _ct15_arg_m_in = _ct15_arg_data.shape[-1]
  _ct15_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct15_arg_m_in
  )
  _ct15_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct15_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct15_arg_r)
  )
  _ct15_arg_moduli = getattr(ct14, "moduli", v0.q_towers)
  if isinstance(_ct15_arg_moduli, (int, np.integer)):
    _ct15_arg_moduli = [int(_ct15_arg_moduli)]
  ct15_arg = Polynomial(
      {
          "batch": _ct15_arg_data.shape[0],
          "num_elements": _ct15_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct15_arg_m,
          "precision": 32,
          "degree_layout": (_ct15_arg_r, _ct15_arg_c),
      },
      {"moduli": list(_ct15_arg_moduli)[:_ct15_arg_m]},
  )
  ct15_arg.polynomial = _ct15_arg_data.reshape(
      _ct15_arg_data.shape[0],
      _ct15_arg_data.shape[1],
      _ct15_arg_r,
      _ct15_arg_c,
      _ct15_arg_m_in,
  )[..., :_ct15_arg_m].copy()
  ct15_arg.batch = ct15_arg.polynomial.shape[0]
  ct15_arg.num_elements = ct15_arg.polynomial.shape[1]
  ct15_arg.num_moduli = _ct15_arg_m
  ct15_arg.degree_layout = (_ct15_arg_r, _ct15_arg_c)
  ct15_arg.r = _ct15_arg_r
  ct15_arg.c = _ct15_arg_c
  ct15_arg.moduli = list(_ct15_arg_moduli)[:_ct15_arg_m]
  ct15_arg.moduli_array = jnp.array(
      ct15_arg.moduli, dtype=getattr(ct15_arg, "modulus_dtype", jnp.uint32)
  )
  ct15_raw = v0.he_rot[v0.max_level, 6].rotate(ct15_arg)
  _ct15_data = (
      ct15_raw.polynomial if hasattr(ct15_raw, "polynomial") else ct15_raw
  )
  _ct15_m_in = _ct15_data.shape[-1]
  _ct15_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct15_m_in
  )
  _ct15_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct15_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct15_r)
  )
  _ct15_moduli = getattr(ct15_raw, "moduli", v0.q_towers)
  if isinstance(_ct15_moduli, (int, np.integer)):
    _ct15_moduli = [int(_ct15_moduli)]
  ct15 = Polynomial(
      {
          "batch": _ct15_data.shape[0],
          "num_elements": _ct15_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct15_m,
          "precision": 32,
          "degree_layout": (_ct15_r, _ct15_c),
      },
      {"moduli": list(_ct15_moduli)[:_ct15_m]},
  )
  ct15.polynomial = _ct15_data.reshape(
      _ct15_data.shape[0], _ct15_data.shape[1], _ct15_r, _ct15_c, _ct15_m_in
  )[..., :_ct15_m].copy()
  ct15.batch = ct15.polynomial.shape[0]
  ct15.num_elements = ct15.polynomial.shape[1]
  ct15.num_moduli = _ct15_m
  ct15.degree_layout = (_ct15_r, _ct15_c)
  ct15.r = _ct15_r
  ct15.c = _ct15_c
  ct15.moduli = list(_ct15_moduli)[:_ct15_m]
  ct15.moduli_array = jnp.array(
      ct15.moduli, dtype=getattr(ct15, "modulus_dtype", jnp.uint32)
  )
  _ct16_data = ct1.polynomial if hasattr(ct1, "polynomial") else ct1
  _ct16_m_in = _ct16_data.shape[-1]
  _ct16_m = _ct16_m_in
  _ct16_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct16_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct16_r)
  )
  _ct16_moduli = getattr(ct1, "moduli", v0.q_towers)
  if isinstance(_ct16_moduli, (int, np.integer)):
    _ct16_moduli = [int(_ct16_moduli)]
  ct16 = Polynomial(
      {
          "batch": _ct16_data.shape[0],
          "num_elements": _ct16_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct16_m,
          "precision": 32,
          "degree_layout": (_ct16_r, _ct16_c),
      },
      {"moduli": list(_ct16_moduli)[:_ct16_m]},
  )
  ct16.polynomial = _ct16_data.reshape(
      _ct16_data.shape[0], _ct16_data.shape[1], _ct16_r, _ct16_c, _ct16_m_in
  )[..., :_ct16_m].copy()
  ct16.batch = ct16.polynomial.shape[0]
  ct16.num_elements = ct16.polynomial.shape[1]
  ct16.num_moduli = _ct16_m
  ct16.degree_layout = (_ct16_r, _ct16_c)
  ct16.r = _ct16_r
  ct16.c = _ct16_c
  ct16.moduli = list(_ct16_moduli)[:_ct16_m]
  ct16.moduli_array = jnp.array(
      ct16.moduli, dtype=getattr(ct16, "modulus_dtype", jnp.uint32)
  )
  _ct16_rhs_data = ct3.polynomial if hasattr(ct3, "polynomial") else ct3
  _ct16_rhs_m_in = _ct16_rhs_data.shape[-1]
  _ct16_rhs_m = _ct16_rhs_m_in
  _ct16_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct16_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct16_rhs_r)
  )
  _ct16_rhs_moduli = getattr(ct3, "moduli", v0.q_towers)
  if isinstance(_ct16_rhs_moduli, (int, np.integer)):
    _ct16_rhs_moduli = [int(_ct16_rhs_moduli)]
  ct16_rhs = Polynomial(
      {
          "batch": _ct16_rhs_data.shape[0],
          "num_elements": _ct16_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct16_rhs_m,
          "precision": 32,
          "degree_layout": (_ct16_rhs_r, _ct16_rhs_c),
      },
      {"moduli": list(_ct16_rhs_moduli)[:_ct16_rhs_m]},
  )
  ct16_rhs.polynomial = _ct16_rhs_data.reshape(
      _ct16_rhs_data.shape[0],
      _ct16_rhs_data.shape[1],
      _ct16_rhs_r,
      _ct16_rhs_c,
      _ct16_rhs_m_in,
  )[..., :_ct16_rhs_m].copy()
  ct16_rhs.batch = ct16_rhs.polynomial.shape[0]
  ct16_rhs.num_elements = ct16_rhs.polynomial.shape[1]
  ct16_rhs.num_moduli = _ct16_rhs_m
  ct16_rhs.degree_layout = (_ct16_rhs_r, _ct16_rhs_c)
  ct16_rhs.r = _ct16_rhs_r
  ct16_rhs.c = _ct16_rhs_c
  ct16_rhs.moduli = list(_ct16_rhs_moduli)[:_ct16_rhs_m]
  ct16_rhs.moduli_array = jnp.array(
      ct16_rhs.moduli, dtype=getattr(ct16_rhs, "modulus_dtype", jnp.uint32)
  )
  ct16.add(ct16_rhs)
  _moduli = jnp.array(ct16.moduli, dtype=jnp.uint32)
  ct16.polynomial = jnp.where(
      ct16.polynomial >= _moduli, ct16.polynomial - _moduli, ct16.polynomial
  )
  _ct17_data = ct5.polynomial if hasattr(ct5, "polynomial") else ct5
  _ct17_m_in = _ct17_data.shape[-1]
  _ct17_m = _ct17_m_in
  _ct17_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct17_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct17_r)
  )
  _ct17_moduli = getattr(ct5, "moduli", v0.q_towers)
  if isinstance(_ct17_moduli, (int, np.integer)):
    _ct17_moduli = [int(_ct17_moduli)]
  ct17 = Polynomial(
      {
          "batch": _ct17_data.shape[0],
          "num_elements": _ct17_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct17_m,
          "precision": 32,
          "degree_layout": (_ct17_r, _ct17_c),
      },
      {"moduli": list(_ct17_moduli)[:_ct17_m]},
  )
  ct17.polynomial = _ct17_data.reshape(
      _ct17_data.shape[0], _ct17_data.shape[1], _ct17_r, _ct17_c, _ct17_m_in
  )[..., :_ct17_m].copy()
  ct17.batch = ct17.polynomial.shape[0]
  ct17.num_elements = ct17.polynomial.shape[1]
  ct17.num_moduli = _ct17_m
  ct17.degree_layout = (_ct17_r, _ct17_c)
  ct17.r = _ct17_r
  ct17.c = _ct17_c
  ct17.moduli = list(_ct17_moduli)[:_ct17_m]
  ct17.moduli_array = jnp.array(
      ct17.moduli, dtype=getattr(ct17, "modulus_dtype", jnp.uint32)
  )
  _ct17_rhs_data = ct11.polynomial if hasattr(ct11, "polynomial") else ct11
  _ct17_rhs_m_in = _ct17_rhs_data.shape[-1]
  _ct17_rhs_m = _ct17_rhs_m_in
  _ct17_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct17_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct17_rhs_r)
  )
  _ct17_rhs_moduli = getattr(ct11, "moduli", v0.q_towers)
  if isinstance(_ct17_rhs_moduli, (int, np.integer)):
    _ct17_rhs_moduli = [int(_ct17_rhs_moduli)]
  ct17_rhs = Polynomial(
      {
          "batch": _ct17_rhs_data.shape[0],
          "num_elements": _ct17_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct17_rhs_m,
          "precision": 32,
          "degree_layout": (_ct17_rhs_r, _ct17_rhs_c),
      },
      {"moduli": list(_ct17_rhs_moduli)[:_ct17_rhs_m]},
  )
  ct17_rhs.polynomial = _ct17_rhs_data.reshape(
      _ct17_rhs_data.shape[0],
      _ct17_rhs_data.shape[1],
      _ct17_rhs_r,
      _ct17_rhs_c,
      _ct17_rhs_m_in,
  )[..., :_ct17_rhs_m].copy()
  ct17_rhs.batch = ct17_rhs.polynomial.shape[0]
  ct17_rhs.num_elements = ct17_rhs.polynomial.shape[1]
  ct17_rhs.num_moduli = _ct17_rhs_m
  ct17_rhs.degree_layout = (_ct17_rhs_r, _ct17_rhs_c)
  ct17_rhs.r = _ct17_rhs_r
  ct17_rhs.c = _ct17_rhs_c
  ct17_rhs.moduli = list(_ct17_rhs_moduli)[:_ct17_rhs_m]
  ct17_rhs.moduli_array = jnp.array(
      ct17_rhs.moduli, dtype=getattr(ct17_rhs, "modulus_dtype", jnp.uint32)
  )
  ct17.add(ct17_rhs)
  _moduli = jnp.array(ct17.moduli, dtype=jnp.uint32)
  ct17.polynomial = jnp.where(
      ct17.polynomial >= _moduli, ct17.polynomial - _moduli, ct17.polynomial
  )
  _ct18_data = ct17.polynomial if hasattr(ct17, "polynomial") else ct17
  _ct18_m_in = _ct18_data.shape[-1]
  _ct18_m = _ct18_m_in
  _ct18_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct18_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct18_r)
  )
  _ct18_moduli = getattr(ct17, "moduli", v0.q_towers)
  if isinstance(_ct18_moduli, (int, np.integer)):
    _ct18_moduli = [int(_ct18_moduli)]
  ct18 = Polynomial(
      {
          "batch": _ct18_data.shape[0],
          "num_elements": _ct18_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct18_m,
          "precision": 32,
          "degree_layout": (_ct18_r, _ct18_c),
      },
      {"moduli": list(_ct18_moduli)[:_ct18_m]},
  )
  ct18.polynomial = _ct18_data.reshape(
      _ct18_data.shape[0], _ct18_data.shape[1], _ct18_r, _ct18_c, _ct18_m_in
  )[..., :_ct18_m].copy()
  ct18.batch = ct18.polynomial.shape[0]
  ct18.num_elements = ct18.polynomial.shape[1]
  ct18.num_moduli = _ct18_m
  ct18.degree_layout = (_ct18_r, _ct18_c)
  ct18.r = _ct18_r
  ct18.c = _ct18_c
  ct18.moduli = list(_ct18_moduli)[:_ct18_m]
  ct18.moduli_array = jnp.array(
      ct18.moduli, dtype=getattr(ct18, "modulus_dtype", jnp.uint32)
  )
  _ct18_rhs_data = ct15.polynomial if hasattr(ct15, "polynomial") else ct15
  _ct18_rhs_m_in = _ct18_rhs_data.shape[-1]
  _ct18_rhs_m = _ct18_rhs_m_in
  _ct18_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct18_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct18_rhs_r)
  )
  _ct18_rhs_moduli = getattr(ct15, "moduli", v0.q_towers)
  if isinstance(_ct18_rhs_moduli, (int, np.integer)):
    _ct18_rhs_moduli = [int(_ct18_rhs_moduli)]
  ct18_rhs = Polynomial(
      {
          "batch": _ct18_rhs_data.shape[0],
          "num_elements": _ct18_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct18_rhs_m,
          "precision": 32,
          "degree_layout": (_ct18_rhs_r, _ct18_rhs_c),
      },
      {"moduli": list(_ct18_rhs_moduli)[:_ct18_rhs_m]},
  )
  ct18_rhs.polynomial = _ct18_rhs_data.reshape(
      _ct18_rhs_data.shape[0],
      _ct18_rhs_data.shape[1],
      _ct18_rhs_r,
      _ct18_rhs_c,
      _ct18_rhs_m_in,
  )[..., :_ct18_rhs_m].copy()
  ct18_rhs.batch = ct18_rhs.polynomial.shape[0]
  ct18_rhs.num_elements = ct18_rhs.polynomial.shape[1]
  ct18_rhs.num_moduli = _ct18_rhs_m
  ct18_rhs.degree_layout = (_ct18_rhs_r, _ct18_rhs_c)
  ct18_rhs.r = _ct18_rhs_r
  ct18_rhs.c = _ct18_rhs_c
  ct18_rhs.moduli = list(_ct18_rhs_moduli)[:_ct18_rhs_m]
  ct18_rhs.moduli_array = jnp.array(
      ct18_rhs.moduli, dtype=getattr(ct18_rhs, "modulus_dtype", jnp.uint32)
  )
  ct18.add(ct18_rhs)
  _moduli = jnp.array(ct18.moduli, dtype=jnp.uint32)
  ct18.polynomial = jnp.where(
      ct18.polynomial >= _moduli, ct18.polynomial - _moduli, ct18.polynomial
  )
  _ct19_data = ct16.polynomial if hasattr(ct16, "polynomial") else ct16
  _ct19_m_in = _ct19_data.shape[-1]
  _ct19_m = _ct19_m_in
  _ct19_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct19_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct19_r)
  )
  _ct19_moduli = getattr(ct16, "moduli", v0.q_towers)
  if isinstance(_ct19_moduli, (int, np.integer)):
    _ct19_moduli = [int(_ct19_moduli)]
  ct19 = Polynomial(
      {
          "batch": _ct19_data.shape[0],
          "num_elements": _ct19_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct19_m,
          "precision": 32,
          "degree_layout": (_ct19_r, _ct19_c),
      },
      {"moduli": list(_ct19_moduli)[:_ct19_m]},
  )
  ct19.polynomial = _ct19_data.reshape(
      _ct19_data.shape[0], _ct19_data.shape[1], _ct19_r, _ct19_c, _ct19_m_in
  )[..., :_ct19_m].copy()
  ct19.batch = ct19.polynomial.shape[0]
  ct19.num_elements = ct19.polynomial.shape[1]
  ct19.num_moduli = _ct19_m
  ct19.degree_layout = (_ct19_r, _ct19_c)
  ct19.r = _ct19_r
  ct19.c = _ct19_c
  ct19.moduli = list(_ct19_moduli)[:_ct19_m]
  ct19.moduli_array = jnp.array(
      ct19.moduli, dtype=getattr(ct19, "modulus_dtype", jnp.uint32)
  )
  _ct19_rhs_data = ct18.polynomial if hasattr(ct18, "polynomial") else ct18
  _ct19_rhs_m_in = _ct19_rhs_data.shape[-1]
  _ct19_rhs_m = _ct19_rhs_m_in
  _ct19_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct19_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct19_rhs_r)
  )
  _ct19_rhs_moduli = getattr(ct18, "moduli", v0.q_towers)
  if isinstance(_ct19_rhs_moduli, (int, np.integer)):
    _ct19_rhs_moduli = [int(_ct19_rhs_moduli)]
  ct19_rhs = Polynomial(
      {
          "batch": _ct19_rhs_data.shape[0],
          "num_elements": _ct19_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct19_rhs_m,
          "precision": 32,
          "degree_layout": (_ct19_rhs_r, _ct19_rhs_c),
      },
      {"moduli": list(_ct19_rhs_moduli)[:_ct19_rhs_m]},
  )
  ct19_rhs.polynomial = _ct19_rhs_data.reshape(
      _ct19_rhs_data.shape[0],
      _ct19_rhs_data.shape[1],
      _ct19_rhs_r,
      _ct19_rhs_c,
      _ct19_rhs_m_in,
  )[..., :_ct19_rhs_m].copy()
  ct19_rhs.batch = ct19_rhs.polynomial.shape[0]
  ct19_rhs.num_elements = ct19_rhs.polynomial.shape[1]
  ct19_rhs.num_moduli = _ct19_rhs_m
  ct19_rhs.degree_layout = (_ct19_rhs_r, _ct19_rhs_c)
  ct19_rhs.r = _ct19_rhs_r
  ct19_rhs.c = _ct19_rhs_c
  ct19_rhs.moduli = list(_ct19_rhs_moduli)[:_ct19_rhs_m]
  ct19_rhs.moduli_array = jnp.array(
      ct19_rhs.moduli, dtype=getattr(ct19_rhs, "modulus_dtype", jnp.uint32)
  )
  ct19.add(ct19_rhs)
  _moduli = jnp.array(ct19.moduli, dtype=jnp.uint32)
  ct19.polynomial = jnp.where(
      ct19.polynomial >= _moduli, ct19.polynomial - _moduli, ct19.polynomial
  )
  _ct20_arg_data = ct19.polynomial if hasattr(ct19, "polynomial") else ct19
  _ct20_arg_m_in = _ct20_arg_data.shape[-1]
  _ct20_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct20_arg_m_in
  )
  _ct20_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct20_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct20_arg_r)
  )
  _ct20_arg_moduli = getattr(ct19, "moduli", v0.q_towers)
  if isinstance(_ct20_arg_moduli, (int, np.integer)):
    _ct20_arg_moduli = [int(_ct20_arg_moduli)]
  ct20_arg = Polynomial(
      {
          "batch": _ct20_arg_data.shape[0],
          "num_elements": _ct20_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct20_arg_m,
          "precision": 32,
          "degree_layout": (_ct20_arg_r, _ct20_arg_c),
      },
      {"moduli": list(_ct20_arg_moduli)[:_ct20_arg_m]},
  )
  ct20_arg.polynomial = _ct20_arg_data.reshape(
      _ct20_arg_data.shape[0],
      _ct20_arg_data.shape[1],
      _ct20_arg_r,
      _ct20_arg_c,
      _ct20_arg_m_in,
  )[..., :_ct20_arg_m].copy()
  ct20_arg.batch = ct20_arg.polynomial.shape[0]
  ct20_arg.num_elements = ct20_arg.polynomial.shape[1]
  ct20_arg.num_moduli = _ct20_arg_m
  ct20_arg.degree_layout = (_ct20_arg_r, _ct20_arg_c)
  ct20_arg.r = _ct20_arg_r
  ct20_arg.c = _ct20_arg_c
  ct20_arg.moduli = list(_ct20_arg_moduli)[:_ct20_arg_m]
  ct20_arg.moduli_array = jnp.array(
      ct20_arg.moduli, dtype=getattr(ct20_arg, "modulus_dtype", jnp.uint32)
  )
  ct20_raw = v0.he_rescale[v0.max_level, v0.max_level - 1](ct20_arg)
  _ct20_data = (
      ct20_raw.polynomial if hasattr(ct20_raw, "polynomial") else ct20_raw
  )
  _ct20_m_in = _ct20_data.shape[-1]
  _ct20_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct20_m_in
  )
  _ct20_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct20_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct20_r)
  )
  _ct20_moduli = getattr(ct20_raw, "moduli", v0.q_towers)
  if isinstance(_ct20_moduli, (int, np.integer)):
    _ct20_moduli = [int(_ct20_moduli)]
  ct20 = Polynomial(
      {
          "batch": _ct20_data.shape[0],
          "num_elements": _ct20_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct20_m,
          "precision": 32,
          "degree_layout": (_ct20_r, _ct20_c),
      },
      {"moduli": list(_ct20_moduli)[:_ct20_m]},
  )
  ct20.polynomial = _ct20_data.reshape(
      _ct20_data.shape[0], _ct20_data.shape[1], _ct20_r, _ct20_c, _ct20_m_in
  )[..., :_ct20_m].copy()
  ct20.batch = ct20.polynomial.shape[0]
  ct20.num_elements = ct20.polynomial.shape[1]
  ct20.num_moduli = _ct20_m
  ct20.degree_layout = (_ct20_r, _ct20_c)
  ct20.r = _ct20_r
  ct20.c = _ct20_c
  ct20.moduli = list(_ct20_moduli)[:_ct20_m]
  ct20.moduli_array = jnp.array(
      ct20.moduli, dtype=getattr(ct20, "modulus_dtype", jnp.uint32)
  )
  _ct21_arg_data = ct20.polynomial if hasattr(ct20, "polynomial") else ct20
  _ct21_arg_m_in = _ct21_arg_data.shape[-1]
  _ct21_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct21_arg_m_in
  )
  _ct21_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct21_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct21_arg_r)
  )
  _ct21_arg_moduli = getattr(ct20, "moduli", v0.q_towers)
  if isinstance(_ct21_arg_moduli, (int, np.integer)):
    _ct21_arg_moduli = [int(_ct21_arg_moduli)]
  ct21_arg = Polynomial(
      {
          "batch": _ct21_arg_data.shape[0],
          "num_elements": _ct21_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct21_arg_m,
          "precision": 32,
          "degree_layout": (_ct21_arg_r, _ct21_arg_c),
      },
      {"moduli": list(_ct21_arg_moduli)[:_ct21_arg_m]},
  )
  ct21_arg.polynomial = _ct21_arg_data.reshape(
      _ct21_arg_data.shape[0],
      _ct21_arg_data.shape[1],
      _ct21_arg_r,
      _ct21_arg_c,
      _ct21_arg_m_in,
  )[..., :_ct21_arg_m].copy()
  ct21_arg.batch = ct21_arg.polynomial.shape[0]
  ct21_arg.num_elements = ct21_arg.polynomial.shape[1]
  ct21_arg.num_moduli = _ct21_arg_m
  ct21_arg.degree_layout = (_ct21_arg_r, _ct21_arg_c)
  ct21_arg.r = _ct21_arg_r
  ct21_arg.c = _ct21_arg_c
  ct21_arg.moduli = list(_ct21_arg_moduli)[:_ct21_arg_m]
  ct21_arg.moduli_array = jnp.array(
      ct21_arg.moduli, dtype=getattr(ct21_arg, "modulus_dtype", jnp.uint32)
  )
  ct21_pt_ntt = (
      pt8.polynomial[0, 0, :, : ct21_arg.polynomial.shape[-1]]
      .reshape(ct21_arg.r, ct21_arg.c, ct21_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct21_ptct = v0.ptct_mul[v0.max_level - 1]
  ct21_ptct.set_plaintext(ct21_pt_ntt)
  ct21_raw = ct21_ptct.mul(ct21_arg, use_bat=False)
  _ct21_data = (
      ct21_raw.polynomial if hasattr(ct21_raw, "polynomial") else ct21_raw
  )
  _ct21_m_in = _ct21_data.shape[-1]
  _ct21_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct21_m_in
  )
  _ct21_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct21_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct21_r)
  )
  _ct21_moduli = getattr(ct21_raw, "moduli", v0.q_towers)
  if isinstance(_ct21_moduli, (int, np.integer)):
    _ct21_moduli = [int(_ct21_moduli)]
  ct21 = Polynomial(
      {
          "batch": _ct21_data.shape[0],
          "num_elements": _ct21_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct21_m,
          "precision": 32,
          "degree_layout": (_ct21_r, _ct21_c),
      },
      {"moduli": list(_ct21_moduli)[:_ct21_m]},
  )
  ct21.polynomial = _ct21_data.reshape(
      _ct21_data.shape[0], _ct21_data.shape[1], _ct21_r, _ct21_c, _ct21_m_in
  )[..., :_ct21_m].copy()
  ct21.batch = ct21.polynomial.shape[0]
  ct21.num_elements = ct21.polynomial.shape[1]
  ct21.num_moduli = _ct21_m
  ct21.degree_layout = (_ct21_r, _ct21_c)
  ct21.r = _ct21_r
  ct21.c = _ct21_c
  ct21.moduli = list(_ct21_moduli)[:_ct21_m]
  ct21.moduli_array = jnp.array(
      ct21.moduli, dtype=getattr(ct21, "modulus_dtype", jnp.uint32)
  )
  _ct22_arg_data = ct19.polynomial if hasattr(ct19, "polynomial") else ct19
  _ct22_arg_m_in = _ct22_arg_data.shape[-1]
  _ct22_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct22_arg_m_in
  )
  _ct22_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct22_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct22_arg_r)
  )
  _ct22_arg_moduli = getattr(ct19, "moduli", v0.q_towers)
  if isinstance(_ct22_arg_moduli, (int, np.integer)):
    _ct22_arg_moduli = [int(_ct22_arg_moduli)]
  ct22_arg = Polynomial(
      {
          "batch": _ct22_arg_data.shape[0],
          "num_elements": _ct22_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct22_arg_m,
          "precision": 32,
          "degree_layout": (_ct22_arg_r, _ct22_arg_c),
      },
      {"moduli": list(_ct22_arg_moduli)[:_ct22_arg_m]},
  )
  ct22_arg.polynomial = _ct22_arg_data.reshape(
      _ct22_arg_data.shape[0],
      _ct22_arg_data.shape[1],
      _ct22_arg_r,
      _ct22_arg_c,
      _ct22_arg_m_in,
  )[..., :_ct22_arg_m].copy()
  ct22_arg.batch = ct22_arg.polynomial.shape[0]
  ct22_arg.num_elements = ct22_arg.polynomial.shape[1]
  ct22_arg.num_moduli = _ct22_arg_m
  ct22_arg.degree_layout = (_ct22_arg_r, _ct22_arg_c)
  ct22_arg.r = _ct22_arg_r
  ct22_arg.c = _ct22_arg_c
  ct22_arg.moduli = list(_ct22_arg_moduli)[:_ct22_arg_m]
  ct22_arg.moduli_array = jnp.array(
      ct22_arg.moduli, dtype=getattr(ct22_arg, "modulus_dtype", jnp.uint32)
  )
  ct22_raw = v0.he_rot[v0.max_level, 1].rotate(ct22_arg)
  _ct22_data = (
      ct22_raw.polynomial if hasattr(ct22_raw, "polynomial") else ct22_raw
  )
  _ct22_m_in = _ct22_data.shape[-1]
  _ct22_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct22_m_in
  )
  _ct22_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct22_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct22_r)
  )
  _ct22_moduli = getattr(ct22_raw, "moduli", v0.q_towers)
  if isinstance(_ct22_moduli, (int, np.integer)):
    _ct22_moduli = [int(_ct22_moduli)]
  ct22 = Polynomial(
      {
          "batch": _ct22_data.shape[0],
          "num_elements": _ct22_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct22_m,
          "precision": 32,
          "degree_layout": (_ct22_r, _ct22_c),
      },
      {"moduli": list(_ct22_moduli)[:_ct22_m]},
  )
  ct22.polynomial = _ct22_data.reshape(
      _ct22_data.shape[0], _ct22_data.shape[1], _ct22_r, _ct22_c, _ct22_m_in
  )[..., :_ct22_m].copy()
  ct22.batch = ct22.polynomial.shape[0]
  ct22.num_elements = ct22.polynomial.shape[1]
  ct22.num_moduli = _ct22_m
  ct22.degree_layout = (_ct22_r, _ct22_c)
  ct22.r = _ct22_r
  ct22.c = _ct22_c
  ct22.moduli = list(_ct22_moduli)[:_ct22_m]
  ct22.moduli_array = jnp.array(
      ct22.moduli, dtype=getattr(ct22, "modulus_dtype", jnp.uint32)
  )
  _ct23_arg_data = ct22.polynomial if hasattr(ct22, "polynomial") else ct22
  _ct23_arg_m_in = _ct23_arg_data.shape[-1]
  _ct23_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct23_arg_m_in
  )
  _ct23_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct23_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct23_arg_r)
  )
  _ct23_arg_moduli = getattr(ct22, "moduli", v0.q_towers)
  if isinstance(_ct23_arg_moduli, (int, np.integer)):
    _ct23_arg_moduli = [int(_ct23_arg_moduli)]
  ct23_arg = Polynomial(
      {
          "batch": _ct23_arg_data.shape[0],
          "num_elements": _ct23_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct23_arg_m,
          "precision": 32,
          "degree_layout": (_ct23_arg_r, _ct23_arg_c),
      },
      {"moduli": list(_ct23_arg_moduli)[:_ct23_arg_m]},
  )
  ct23_arg.polynomial = _ct23_arg_data.reshape(
      _ct23_arg_data.shape[0],
      _ct23_arg_data.shape[1],
      _ct23_arg_r,
      _ct23_arg_c,
      _ct23_arg_m_in,
  )[..., :_ct23_arg_m].copy()
  ct23_arg.batch = ct23_arg.polynomial.shape[0]
  ct23_arg.num_elements = ct23_arg.polynomial.shape[1]
  ct23_arg.num_moduli = _ct23_arg_m
  ct23_arg.degree_layout = (_ct23_arg_r, _ct23_arg_c)
  ct23_arg.r = _ct23_arg_r
  ct23_arg.c = _ct23_arg_c
  ct23_arg.moduli = list(_ct23_arg_moduli)[:_ct23_arg_m]
  ct23_arg.moduli_array = jnp.array(
      ct23_arg.moduli, dtype=getattr(ct23_arg, "modulus_dtype", jnp.uint32)
  )
  ct23_raw = v0.he_rescale[v0.max_level, v0.max_level - 1](ct23_arg)
  _ct23_data = (
      ct23_raw.polynomial if hasattr(ct23_raw, "polynomial") else ct23_raw
  )
  _ct23_m_in = _ct23_data.shape[-1]
  _ct23_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct23_m_in
  )
  _ct23_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct23_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct23_r)
  )
  _ct23_moduli = getattr(ct23_raw, "moduli", v0.q_towers)
  if isinstance(_ct23_moduli, (int, np.integer)):
    _ct23_moduli = [int(_ct23_moduli)]
  ct23 = Polynomial(
      {
          "batch": _ct23_data.shape[0],
          "num_elements": _ct23_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct23_m,
          "precision": 32,
          "degree_layout": (_ct23_r, _ct23_c),
      },
      {"moduli": list(_ct23_moduli)[:_ct23_m]},
  )
  ct23.polynomial = _ct23_data.reshape(
      _ct23_data.shape[0], _ct23_data.shape[1], _ct23_r, _ct23_c, _ct23_m_in
  )[..., :_ct23_m].copy()
  ct23.batch = ct23.polynomial.shape[0]
  ct23.num_elements = ct23.polynomial.shape[1]
  ct23.num_moduli = _ct23_m
  ct23.degree_layout = (_ct23_r, _ct23_c)
  ct23.r = _ct23_r
  ct23.c = _ct23_c
  ct23.moduli = list(_ct23_moduli)[:_ct23_m]
  ct23.moduli_array = jnp.array(
      ct23.moduli, dtype=getattr(ct23, "modulus_dtype", jnp.uint32)
  )
  _ct24_arg_data = ct23.polynomial if hasattr(ct23, "polynomial") else ct23
  _ct24_arg_m_in = _ct24_arg_data.shape[-1]
  _ct24_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct24_arg_m_in
  )
  _ct24_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct24_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct24_arg_r)
  )
  _ct24_arg_moduli = getattr(ct23, "moduli", v0.q_towers)
  if isinstance(_ct24_arg_moduli, (int, np.integer)):
    _ct24_arg_moduli = [int(_ct24_arg_moduli)]
  ct24_arg = Polynomial(
      {
          "batch": _ct24_arg_data.shape[0],
          "num_elements": _ct24_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct24_arg_m,
          "precision": 32,
          "degree_layout": (_ct24_arg_r, _ct24_arg_c),
      },
      {"moduli": list(_ct24_arg_moduli)[:_ct24_arg_m]},
  )
  ct24_arg.polynomial = _ct24_arg_data.reshape(
      _ct24_arg_data.shape[0],
      _ct24_arg_data.shape[1],
      _ct24_arg_r,
      _ct24_arg_c,
      _ct24_arg_m_in,
  )[..., :_ct24_arg_m].copy()
  ct24_arg.batch = ct24_arg.polynomial.shape[0]
  ct24_arg.num_elements = ct24_arg.polynomial.shape[1]
  ct24_arg.num_moduli = _ct24_arg_m
  ct24_arg.degree_layout = (_ct24_arg_r, _ct24_arg_c)
  ct24_arg.r = _ct24_arg_r
  ct24_arg.c = _ct24_arg_c
  ct24_arg.moduli = list(_ct24_arg_moduli)[:_ct24_arg_m]
  ct24_arg.moduli_array = jnp.array(
      ct24_arg.moduli, dtype=getattr(ct24_arg, "modulus_dtype", jnp.uint32)
  )
  ct24_pt_ntt = (
      pt9.polynomial[0, 0, :, : ct24_arg.polynomial.shape[-1]]
      .reshape(ct24_arg.r, ct24_arg.c, ct24_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct24_ptct = v0.ptct_mul[v0.max_level - 1]
  ct24_ptct.set_plaintext(ct24_pt_ntt)
  ct24_raw = ct24_ptct.mul(ct24_arg, use_bat=False)
  _ct24_data = (
      ct24_raw.polynomial if hasattr(ct24_raw, "polynomial") else ct24_raw
  )
  _ct24_m_in = _ct24_data.shape[-1]
  _ct24_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct24_m_in
  )
  _ct24_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct24_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct24_r)
  )
  _ct24_moduli = getattr(ct24_raw, "moduli", v0.q_towers)
  if isinstance(_ct24_moduli, (int, np.integer)):
    _ct24_moduli = [int(_ct24_moduli)]
  ct24 = Polynomial(
      {
          "batch": _ct24_data.shape[0],
          "num_elements": _ct24_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct24_m,
          "precision": 32,
          "degree_layout": (_ct24_r, _ct24_c),
      },
      {"moduli": list(_ct24_moduli)[:_ct24_m]},
  )
  ct24.polynomial = _ct24_data.reshape(
      _ct24_data.shape[0], _ct24_data.shape[1], _ct24_r, _ct24_c, _ct24_m_in
  )[..., :_ct24_m].copy()
  ct24.batch = ct24.polynomial.shape[0]
  ct24.num_elements = ct24.polynomial.shape[1]
  ct24.num_moduli = _ct24_m
  ct24.degree_layout = (_ct24_r, _ct24_c)
  ct24.r = _ct24_r
  ct24.c = _ct24_c
  ct24.moduli = list(_ct24_moduli)[:_ct24_m]
  ct24.moduli_array = jnp.array(
      ct24.moduli, dtype=getattr(ct24, "modulus_dtype", jnp.uint32)
  )
  _ct25_arg_data = ct19.polynomial if hasattr(ct19, "polynomial") else ct19
  _ct25_arg_m_in = _ct25_arg_data.shape[-1]
  _ct25_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct25_arg_m_in
  )
  _ct25_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct25_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct25_arg_r)
  )
  _ct25_arg_moduli = getattr(ct19, "moduli", v0.q_towers)
  if isinstance(_ct25_arg_moduli, (int, np.integer)):
    _ct25_arg_moduli = [int(_ct25_arg_moduli)]
  ct25_arg = Polynomial(
      {
          "batch": _ct25_arg_data.shape[0],
          "num_elements": _ct25_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct25_arg_m,
          "precision": 32,
          "degree_layout": (_ct25_arg_r, _ct25_arg_c),
      },
      {"moduli": list(_ct25_arg_moduli)[:_ct25_arg_m]},
  )
  ct25_arg.polynomial = _ct25_arg_data.reshape(
      _ct25_arg_data.shape[0],
      _ct25_arg_data.shape[1],
      _ct25_arg_r,
      _ct25_arg_c,
      _ct25_arg_m_in,
  )[..., :_ct25_arg_m].copy()
  ct25_arg.batch = ct25_arg.polynomial.shape[0]
  ct25_arg.num_elements = ct25_arg.polynomial.shape[1]
  ct25_arg.num_moduli = _ct25_arg_m
  ct25_arg.degree_layout = (_ct25_arg_r, _ct25_arg_c)
  ct25_arg.r = _ct25_arg_r
  ct25_arg.c = _ct25_arg_c
  ct25_arg.moduli = list(_ct25_arg_moduli)[:_ct25_arg_m]
  ct25_arg.moduli_array = jnp.array(
      ct25_arg.moduli, dtype=getattr(ct25_arg, "modulus_dtype", jnp.uint32)
  )
  ct25_raw = v0.he_rot[v0.max_level, 2].rotate(ct25_arg)
  _ct25_data = (
      ct25_raw.polynomial if hasattr(ct25_raw, "polynomial") else ct25_raw
  )
  _ct25_m_in = _ct25_data.shape[-1]
  _ct25_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct25_m_in
  )
  _ct25_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct25_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct25_r)
  )
  _ct25_moduli = getattr(ct25_raw, "moduli", v0.q_towers)
  if isinstance(_ct25_moduli, (int, np.integer)):
    _ct25_moduli = [int(_ct25_moduli)]
  ct25 = Polynomial(
      {
          "batch": _ct25_data.shape[0],
          "num_elements": _ct25_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct25_m,
          "precision": 32,
          "degree_layout": (_ct25_r, _ct25_c),
      },
      {"moduli": list(_ct25_moduli)[:_ct25_m]},
  )
  ct25.polynomial = _ct25_data.reshape(
      _ct25_data.shape[0], _ct25_data.shape[1], _ct25_r, _ct25_c, _ct25_m_in
  )[..., :_ct25_m].copy()
  ct25.batch = ct25.polynomial.shape[0]
  ct25.num_elements = ct25.polynomial.shape[1]
  ct25.num_moduli = _ct25_m
  ct25.degree_layout = (_ct25_r, _ct25_c)
  ct25.r = _ct25_r
  ct25.c = _ct25_c
  ct25.moduli = list(_ct25_moduli)[:_ct25_m]
  ct25.moduli_array = jnp.array(
      ct25.moduli, dtype=getattr(ct25, "modulus_dtype", jnp.uint32)
  )
  _ct26_arg_data = ct25.polynomial if hasattr(ct25, "polynomial") else ct25
  _ct26_arg_m_in = _ct26_arg_data.shape[-1]
  _ct26_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level)
      if hasattr(v0, "_param_cache")
      else _ct26_arg_m_in
  )
  _ct26_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct26_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct26_arg_r)
  )
  _ct26_arg_moduli = getattr(ct25, "moduli", v0.q_towers)
  if isinstance(_ct26_arg_moduli, (int, np.integer)):
    _ct26_arg_moduli = [int(_ct26_arg_moduli)]
  ct26_arg = Polynomial(
      {
          "batch": _ct26_arg_data.shape[0],
          "num_elements": _ct26_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct26_arg_m,
          "precision": 32,
          "degree_layout": (_ct26_arg_r, _ct26_arg_c),
      },
      {"moduli": list(_ct26_arg_moduli)[:_ct26_arg_m]},
  )
  ct26_arg.polynomial = _ct26_arg_data.reshape(
      _ct26_arg_data.shape[0],
      _ct26_arg_data.shape[1],
      _ct26_arg_r,
      _ct26_arg_c,
      _ct26_arg_m_in,
  )[..., :_ct26_arg_m].copy()
  ct26_arg.batch = ct26_arg.polynomial.shape[0]
  ct26_arg.num_elements = ct26_arg.polynomial.shape[1]
  ct26_arg.num_moduli = _ct26_arg_m
  ct26_arg.degree_layout = (_ct26_arg_r, _ct26_arg_c)
  ct26_arg.r = _ct26_arg_r
  ct26_arg.c = _ct26_arg_c
  ct26_arg.moduli = list(_ct26_arg_moduli)[:_ct26_arg_m]
  ct26_arg.moduli_array = jnp.array(
      ct26_arg.moduli, dtype=getattr(ct26_arg, "modulus_dtype", jnp.uint32)
  )
  ct26_raw = v0.he_rescale[v0.max_level, v0.max_level - 1](ct26_arg)
  _ct26_data = (
      ct26_raw.polynomial if hasattr(ct26_raw, "polynomial") else ct26_raw
  )
  _ct26_m_in = _ct26_data.shape[-1]
  _ct26_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct26_m_in
  )
  _ct26_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct26_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct26_r)
  )
  _ct26_moduli = getattr(ct26_raw, "moduli", v0.q_towers)
  if isinstance(_ct26_moduli, (int, np.integer)):
    _ct26_moduli = [int(_ct26_moduli)]
  ct26 = Polynomial(
      {
          "batch": _ct26_data.shape[0],
          "num_elements": _ct26_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct26_m,
          "precision": 32,
          "degree_layout": (_ct26_r, _ct26_c),
      },
      {"moduli": list(_ct26_moduli)[:_ct26_m]},
  )
  ct26.polynomial = _ct26_data.reshape(
      _ct26_data.shape[0], _ct26_data.shape[1], _ct26_r, _ct26_c, _ct26_m_in
  )[..., :_ct26_m].copy()
  ct26.batch = ct26.polynomial.shape[0]
  ct26.num_elements = ct26.polynomial.shape[1]
  ct26.num_moduli = _ct26_m
  ct26.degree_layout = (_ct26_r, _ct26_c)
  ct26.r = _ct26_r
  ct26.c = _ct26_c
  ct26.moduli = list(_ct26_moduli)[:_ct26_m]
  ct26.moduli_array = jnp.array(
      ct26.moduli, dtype=getattr(ct26, "modulus_dtype", jnp.uint32)
  )
  _ct27_arg_data = ct26.polynomial if hasattr(ct26, "polynomial") else ct26
  _ct27_arg_m_in = _ct27_arg_data.shape[-1]
  _ct27_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct27_arg_m_in
  )
  _ct27_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct27_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct27_arg_r)
  )
  _ct27_arg_moduli = getattr(ct26, "moduli", v0.q_towers)
  if isinstance(_ct27_arg_moduli, (int, np.integer)):
    _ct27_arg_moduli = [int(_ct27_arg_moduli)]
  ct27_arg = Polynomial(
      {
          "batch": _ct27_arg_data.shape[0],
          "num_elements": _ct27_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct27_arg_m,
          "precision": 32,
          "degree_layout": (_ct27_arg_r, _ct27_arg_c),
      },
      {"moduli": list(_ct27_arg_moduli)[:_ct27_arg_m]},
  )
  ct27_arg.polynomial = _ct27_arg_data.reshape(
      _ct27_arg_data.shape[0],
      _ct27_arg_data.shape[1],
      _ct27_arg_r,
      _ct27_arg_c,
      _ct27_arg_m_in,
  )[..., :_ct27_arg_m].copy()
  ct27_arg.batch = ct27_arg.polynomial.shape[0]
  ct27_arg.num_elements = ct27_arg.polynomial.shape[1]
  ct27_arg.num_moduli = _ct27_arg_m
  ct27_arg.degree_layout = (_ct27_arg_r, _ct27_arg_c)
  ct27_arg.r = _ct27_arg_r
  ct27_arg.c = _ct27_arg_c
  ct27_arg.moduli = list(_ct27_arg_moduli)[:_ct27_arg_m]
  ct27_arg.moduli_array = jnp.array(
      ct27_arg.moduli, dtype=getattr(ct27_arg, "modulus_dtype", jnp.uint32)
  )
  ct27_pt_ntt = (
      pt10.polynomial[0, 0, :, : ct27_arg.polynomial.shape[-1]]
      .reshape(ct27_arg.r, ct27_arg.c, ct27_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct27_ptct = v0.ptct_mul[v0.max_level - 1]
  ct27_ptct.set_plaintext(ct27_pt_ntt)
  ct27_raw = ct27_ptct.mul(ct27_arg, use_bat=False)
  _ct27_data = (
      ct27_raw.polynomial if hasattr(ct27_raw, "polynomial") else ct27_raw
  )
  _ct27_m_in = _ct27_data.shape[-1]
  _ct27_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct27_m_in
  )
  _ct27_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct27_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct27_r)
  )
  _ct27_moduli = getattr(ct27_raw, "moduli", v0.q_towers)
  if isinstance(_ct27_moduli, (int, np.integer)):
    _ct27_moduli = [int(_ct27_moduli)]
  ct27 = Polynomial(
      {
          "batch": _ct27_data.shape[0],
          "num_elements": _ct27_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct27_m,
          "precision": 32,
          "degree_layout": (_ct27_r, _ct27_c),
      },
      {"moduli": list(_ct27_moduli)[:_ct27_m]},
  )
  ct27.polynomial = _ct27_data.reshape(
      _ct27_data.shape[0], _ct27_data.shape[1], _ct27_r, _ct27_c, _ct27_m_in
  )[..., :_ct27_m].copy()
  ct27.batch = ct27.polynomial.shape[0]
  ct27.num_elements = ct27.polynomial.shape[1]
  ct27.num_moduli = _ct27_m
  ct27.degree_layout = (_ct27_r, _ct27_c)
  ct27.r = _ct27_r
  ct27.c = _ct27_c
  ct27.moduli = list(_ct27_moduli)[:_ct27_m]
  ct27.moduli_array = jnp.array(
      ct27.moduli, dtype=getattr(ct27, "modulus_dtype", jnp.uint32)
  )
  _ct28_arg_data = ct20.polynomial if hasattr(ct20, "polynomial") else ct20
  _ct28_arg_m_in = _ct28_arg_data.shape[-1]
  _ct28_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct28_arg_m_in
  )
  _ct28_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct28_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct28_arg_r)
  )
  _ct28_arg_moduli = getattr(ct20, "moduli", v0.q_towers)
  if isinstance(_ct28_arg_moduli, (int, np.integer)):
    _ct28_arg_moduli = [int(_ct28_arg_moduli)]
  ct28_arg = Polynomial(
      {
          "batch": _ct28_arg_data.shape[0],
          "num_elements": _ct28_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct28_arg_m,
          "precision": 32,
          "degree_layout": (_ct28_arg_r, _ct28_arg_c),
      },
      {"moduli": list(_ct28_arg_moduli)[:_ct28_arg_m]},
  )
  ct28_arg.polynomial = _ct28_arg_data.reshape(
      _ct28_arg_data.shape[0],
      _ct28_arg_data.shape[1],
      _ct28_arg_r,
      _ct28_arg_c,
      _ct28_arg_m_in,
  )[..., :_ct28_arg_m].copy()
  ct28_arg.batch = ct28_arg.polynomial.shape[0]
  ct28_arg.num_elements = ct28_arg.polynomial.shape[1]
  ct28_arg.num_moduli = _ct28_arg_m
  ct28_arg.degree_layout = (_ct28_arg_r, _ct28_arg_c)
  ct28_arg.r = _ct28_arg_r
  ct28_arg.c = _ct28_arg_c
  ct28_arg.moduli = list(_ct28_arg_moduli)[:_ct28_arg_m]
  ct28_arg.moduli_array = jnp.array(
      ct28_arg.moduli, dtype=getattr(ct28_arg, "modulus_dtype", jnp.uint32)
  )
  ct28_pt_ntt = (
      pt11.polynomial[0, 0, :, : ct28_arg.polynomial.shape[-1]]
      .reshape(ct28_arg.r, ct28_arg.c, ct28_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct28_ptct = v0.ptct_mul[v0.max_level - 1]
  ct28_ptct.set_plaintext(ct28_pt_ntt)
  ct28_raw = ct28_ptct.mul(ct28_arg, use_bat=False)
  _ct28_data = (
      ct28_raw.polynomial if hasattr(ct28_raw, "polynomial") else ct28_raw
  )
  _ct28_m_in = _ct28_data.shape[-1]
  _ct28_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct28_m_in
  )
  _ct28_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct28_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct28_r)
  )
  _ct28_moduli = getattr(ct28_raw, "moduli", v0.q_towers)
  if isinstance(_ct28_moduli, (int, np.integer)):
    _ct28_moduli = [int(_ct28_moduli)]
  ct28 = Polynomial(
      {
          "batch": _ct28_data.shape[0],
          "num_elements": _ct28_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct28_m,
          "precision": 32,
          "degree_layout": (_ct28_r, _ct28_c),
      },
      {"moduli": list(_ct28_moduli)[:_ct28_m]},
  )
  ct28.polynomial = _ct28_data.reshape(
      _ct28_data.shape[0], _ct28_data.shape[1], _ct28_r, _ct28_c, _ct28_m_in
  )[..., :_ct28_m].copy()
  ct28.batch = ct28.polynomial.shape[0]
  ct28.num_elements = ct28.polynomial.shape[1]
  ct28.num_moduli = _ct28_m
  ct28.degree_layout = (_ct28_r, _ct28_c)
  ct28.r = _ct28_r
  ct28.c = _ct28_c
  ct28.moduli = list(_ct28_moduli)[:_ct28_m]
  ct28.moduli_array = jnp.array(
      ct28.moduli, dtype=getattr(ct28, "modulus_dtype", jnp.uint32)
  )
  _ct29_arg_data = ct23.polynomial if hasattr(ct23, "polynomial") else ct23
  _ct29_arg_m_in = _ct29_arg_data.shape[-1]
  _ct29_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct29_arg_m_in
  )
  _ct29_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct29_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct29_arg_r)
  )
  _ct29_arg_moduli = getattr(ct23, "moduli", v0.q_towers)
  if isinstance(_ct29_arg_moduli, (int, np.integer)):
    _ct29_arg_moduli = [int(_ct29_arg_moduli)]
  ct29_arg = Polynomial(
      {
          "batch": _ct29_arg_data.shape[0],
          "num_elements": _ct29_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct29_arg_m,
          "precision": 32,
          "degree_layout": (_ct29_arg_r, _ct29_arg_c),
      },
      {"moduli": list(_ct29_arg_moduli)[:_ct29_arg_m]},
  )
  ct29_arg.polynomial = _ct29_arg_data.reshape(
      _ct29_arg_data.shape[0],
      _ct29_arg_data.shape[1],
      _ct29_arg_r,
      _ct29_arg_c,
      _ct29_arg_m_in,
  )[..., :_ct29_arg_m].copy()
  ct29_arg.batch = ct29_arg.polynomial.shape[0]
  ct29_arg.num_elements = ct29_arg.polynomial.shape[1]
  ct29_arg.num_moduli = _ct29_arg_m
  ct29_arg.degree_layout = (_ct29_arg_r, _ct29_arg_c)
  ct29_arg.r = _ct29_arg_r
  ct29_arg.c = _ct29_arg_c
  ct29_arg.moduli = list(_ct29_arg_moduli)[:_ct29_arg_m]
  ct29_arg.moduli_array = jnp.array(
      ct29_arg.moduli, dtype=getattr(ct29_arg, "modulus_dtype", jnp.uint32)
  )
  ct29_pt_ntt = (
      pt12.polynomial[0, 0, :, : ct29_arg.polynomial.shape[-1]]
      .reshape(ct29_arg.r, ct29_arg.c, ct29_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct29_ptct = v0.ptct_mul[v0.max_level - 1]
  ct29_ptct.set_plaintext(ct29_pt_ntt)
  ct29_raw = ct29_ptct.mul(ct29_arg, use_bat=False)
  _ct29_data = (
      ct29_raw.polynomial if hasattr(ct29_raw, "polynomial") else ct29_raw
  )
  _ct29_m_in = _ct29_data.shape[-1]
  _ct29_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct29_m_in
  )
  _ct29_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct29_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct29_r)
  )
  _ct29_moduli = getattr(ct29_raw, "moduli", v0.q_towers)
  if isinstance(_ct29_moduli, (int, np.integer)):
    _ct29_moduli = [int(_ct29_moduli)]
  ct29 = Polynomial(
      {
          "batch": _ct29_data.shape[0],
          "num_elements": _ct29_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct29_m,
          "precision": 32,
          "degree_layout": (_ct29_r, _ct29_c),
      },
      {"moduli": list(_ct29_moduli)[:_ct29_m]},
  )
  ct29.polynomial = _ct29_data.reshape(
      _ct29_data.shape[0], _ct29_data.shape[1], _ct29_r, _ct29_c, _ct29_m_in
  )[..., :_ct29_m].copy()
  ct29.batch = ct29.polynomial.shape[0]
  ct29.num_elements = ct29.polynomial.shape[1]
  ct29.num_moduli = _ct29_m
  ct29.degree_layout = (_ct29_r, _ct29_c)
  ct29.r = _ct29_r
  ct29.c = _ct29_c
  ct29.moduli = list(_ct29_moduli)[:_ct29_m]
  ct29.moduli_array = jnp.array(
      ct29.moduli, dtype=getattr(ct29, "modulus_dtype", jnp.uint32)
  )
  _ct30_arg_data = ct26.polynomial if hasattr(ct26, "polynomial") else ct26
  _ct30_arg_m_in = _ct30_arg_data.shape[-1]
  _ct30_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct30_arg_m_in
  )
  _ct30_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct30_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct30_arg_r)
  )
  _ct30_arg_moduli = getattr(ct26, "moduli", v0.q_towers)
  if isinstance(_ct30_arg_moduli, (int, np.integer)):
    _ct30_arg_moduli = [int(_ct30_arg_moduli)]
  ct30_arg = Polynomial(
      {
          "batch": _ct30_arg_data.shape[0],
          "num_elements": _ct30_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct30_arg_m,
          "precision": 32,
          "degree_layout": (_ct30_arg_r, _ct30_arg_c),
      },
      {"moduli": list(_ct30_arg_moduli)[:_ct30_arg_m]},
  )
  ct30_arg.polynomial = _ct30_arg_data.reshape(
      _ct30_arg_data.shape[0],
      _ct30_arg_data.shape[1],
      _ct30_arg_r,
      _ct30_arg_c,
      _ct30_arg_m_in,
  )[..., :_ct30_arg_m].copy()
  ct30_arg.batch = ct30_arg.polynomial.shape[0]
  ct30_arg.num_elements = ct30_arg.polynomial.shape[1]
  ct30_arg.num_moduli = _ct30_arg_m
  ct30_arg.degree_layout = (_ct30_arg_r, _ct30_arg_c)
  ct30_arg.r = _ct30_arg_r
  ct30_arg.c = _ct30_arg_c
  ct30_arg.moduli = list(_ct30_arg_moduli)[:_ct30_arg_m]
  ct30_arg.moduli_array = jnp.array(
      ct30_arg.moduli, dtype=getattr(ct30_arg, "modulus_dtype", jnp.uint32)
  )
  ct30_pt_ntt = (
      pt13.polynomial[0, 0, :, : ct30_arg.polynomial.shape[-1]]
      .reshape(ct30_arg.r, ct30_arg.c, ct30_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct30_ptct = v0.ptct_mul[v0.max_level - 1]
  ct30_ptct.set_plaintext(ct30_pt_ntt)
  ct30_raw = ct30_ptct.mul(ct30_arg, use_bat=False)
  _ct30_data = (
      ct30_raw.polynomial if hasattr(ct30_raw, "polynomial") else ct30_raw
  )
  _ct30_m_in = _ct30_data.shape[-1]
  _ct30_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct30_m_in
  )
  _ct30_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct30_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct30_r)
  )
  _ct30_moduli = getattr(ct30_raw, "moduli", v0.q_towers)
  if isinstance(_ct30_moduli, (int, np.integer)):
    _ct30_moduli = [int(_ct30_moduli)]
  ct30 = Polynomial(
      {
          "batch": _ct30_data.shape[0],
          "num_elements": _ct30_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct30_m,
          "precision": 32,
          "degree_layout": (_ct30_r, _ct30_c),
      },
      {"moduli": list(_ct30_moduli)[:_ct30_m]},
  )
  ct30.polynomial = _ct30_data.reshape(
      _ct30_data.shape[0], _ct30_data.shape[1], _ct30_r, _ct30_c, _ct30_m_in
  )[..., :_ct30_m].copy()
  ct30.batch = ct30.polynomial.shape[0]
  ct30.num_elements = ct30.polynomial.shape[1]
  ct30.num_moduli = _ct30_m
  ct30.degree_layout = (_ct30_r, _ct30_c)
  ct30.r = _ct30_r
  ct30.c = _ct30_c
  ct30.moduli = list(_ct30_moduli)[:_ct30_m]
  ct30.moduli_array = jnp.array(
      ct30.moduli, dtype=getattr(ct30, "modulus_dtype", jnp.uint32)
  )
  _ct31_data = ct28.polynomial if hasattr(ct28, "polynomial") else ct28
  _ct31_m_in = _ct31_data.shape[-1]
  _ct31_m = _ct31_m_in
  _ct31_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct31_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct31_r)
  )
  _ct31_moduli = getattr(ct28, "moduli", v0.q_towers)
  if isinstance(_ct31_moduli, (int, np.integer)):
    _ct31_moduli = [int(_ct31_moduli)]
  ct31 = Polynomial(
      {
          "batch": _ct31_data.shape[0],
          "num_elements": _ct31_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct31_m,
          "precision": 32,
          "degree_layout": (_ct31_r, _ct31_c),
      },
      {"moduli": list(_ct31_moduli)[:_ct31_m]},
  )
  ct31.polynomial = _ct31_data.reshape(
      _ct31_data.shape[0], _ct31_data.shape[1], _ct31_r, _ct31_c, _ct31_m_in
  )[..., :_ct31_m].copy()
  ct31.batch = ct31.polynomial.shape[0]
  ct31.num_elements = ct31.polynomial.shape[1]
  ct31.num_moduli = _ct31_m
  ct31.degree_layout = (_ct31_r, _ct31_c)
  ct31.r = _ct31_r
  ct31.c = _ct31_c
  ct31.moduli = list(_ct31_moduli)[:_ct31_m]
  ct31.moduli_array = jnp.array(
      ct31.moduli, dtype=getattr(ct31, "modulus_dtype", jnp.uint32)
  )
  _ct31_rhs_data = ct29.polynomial if hasattr(ct29, "polynomial") else ct29
  _ct31_rhs_m_in = _ct31_rhs_data.shape[-1]
  _ct31_rhs_m = _ct31_rhs_m_in
  _ct31_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct31_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct31_rhs_r)
  )
  _ct31_rhs_moduli = getattr(ct29, "moduli", v0.q_towers)
  if isinstance(_ct31_rhs_moduli, (int, np.integer)):
    _ct31_rhs_moduli = [int(_ct31_rhs_moduli)]
  ct31_rhs = Polynomial(
      {
          "batch": _ct31_rhs_data.shape[0],
          "num_elements": _ct31_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct31_rhs_m,
          "precision": 32,
          "degree_layout": (_ct31_rhs_r, _ct31_rhs_c),
      },
      {"moduli": list(_ct31_rhs_moduli)[:_ct31_rhs_m]},
  )
  ct31_rhs.polynomial = _ct31_rhs_data.reshape(
      _ct31_rhs_data.shape[0],
      _ct31_rhs_data.shape[1],
      _ct31_rhs_r,
      _ct31_rhs_c,
      _ct31_rhs_m_in,
  )[..., :_ct31_rhs_m].copy()
  ct31_rhs.batch = ct31_rhs.polynomial.shape[0]
  ct31_rhs.num_elements = ct31_rhs.polynomial.shape[1]
  ct31_rhs.num_moduli = _ct31_rhs_m
  ct31_rhs.degree_layout = (_ct31_rhs_r, _ct31_rhs_c)
  ct31_rhs.r = _ct31_rhs_r
  ct31_rhs.c = _ct31_rhs_c
  ct31_rhs.moduli = list(_ct31_rhs_moduli)[:_ct31_rhs_m]
  ct31_rhs.moduli_array = jnp.array(
      ct31_rhs.moduli, dtype=getattr(ct31_rhs, "modulus_dtype", jnp.uint32)
  )
  ct31.add(ct31_rhs)
  _moduli = jnp.array(ct31.moduli, dtype=jnp.uint32)
  ct31.polynomial = jnp.where(
      ct31.polynomial >= _moduli, ct31.polynomial - _moduli, ct31.polynomial
  )
  _ct32_data = ct31.polynomial if hasattr(ct31, "polynomial") else ct31
  _ct32_m_in = _ct32_data.shape[-1]
  _ct32_m = _ct32_m_in
  _ct32_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct32_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct32_r)
  )
  _ct32_moduli = getattr(ct31, "moduli", v0.q_towers)
  if isinstance(_ct32_moduli, (int, np.integer)):
    _ct32_moduli = [int(_ct32_moduli)]
  ct32 = Polynomial(
      {
          "batch": _ct32_data.shape[0],
          "num_elements": _ct32_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct32_m,
          "precision": 32,
          "degree_layout": (_ct32_r, _ct32_c),
      },
      {"moduli": list(_ct32_moduli)[:_ct32_m]},
  )
  ct32.polynomial = _ct32_data.reshape(
      _ct32_data.shape[0], _ct32_data.shape[1], _ct32_r, _ct32_c, _ct32_m_in
  )[..., :_ct32_m].copy()
  ct32.batch = ct32.polynomial.shape[0]
  ct32.num_elements = ct32.polynomial.shape[1]
  ct32.num_moduli = _ct32_m
  ct32.degree_layout = (_ct32_r, _ct32_c)
  ct32.r = _ct32_r
  ct32.c = _ct32_c
  ct32.moduli = list(_ct32_moduli)[:_ct32_m]
  ct32.moduli_array = jnp.array(
      ct32.moduli, dtype=getattr(ct32, "modulus_dtype", jnp.uint32)
  )
  _ct32_rhs_data = ct30.polynomial if hasattr(ct30, "polynomial") else ct30
  _ct32_rhs_m_in = _ct32_rhs_data.shape[-1]
  _ct32_rhs_m = _ct32_rhs_m_in
  _ct32_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct32_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct32_rhs_r)
  )
  _ct32_rhs_moduli = getattr(ct30, "moduli", v0.q_towers)
  if isinstance(_ct32_rhs_moduli, (int, np.integer)):
    _ct32_rhs_moduli = [int(_ct32_rhs_moduli)]
  ct32_rhs = Polynomial(
      {
          "batch": _ct32_rhs_data.shape[0],
          "num_elements": _ct32_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct32_rhs_m,
          "precision": 32,
          "degree_layout": (_ct32_rhs_r, _ct32_rhs_c),
      },
      {"moduli": list(_ct32_rhs_moduli)[:_ct32_rhs_m]},
  )
  ct32_rhs.polynomial = _ct32_rhs_data.reshape(
      _ct32_rhs_data.shape[0],
      _ct32_rhs_data.shape[1],
      _ct32_rhs_r,
      _ct32_rhs_c,
      _ct32_rhs_m_in,
  )[..., :_ct32_rhs_m].copy()
  ct32_rhs.batch = ct32_rhs.polynomial.shape[0]
  ct32_rhs.num_elements = ct32_rhs.polynomial.shape[1]
  ct32_rhs.num_moduli = _ct32_rhs_m
  ct32_rhs.degree_layout = (_ct32_rhs_r, _ct32_rhs_c)
  ct32_rhs.r = _ct32_rhs_r
  ct32_rhs.c = _ct32_rhs_c
  ct32_rhs.moduli = list(_ct32_rhs_moduli)[:_ct32_rhs_m]
  ct32_rhs.moduli_array = jnp.array(
      ct32_rhs.moduli, dtype=getattr(ct32_rhs, "modulus_dtype", jnp.uint32)
  )
  ct32.add(ct32_rhs)
  _moduli = jnp.array(ct32.moduli, dtype=jnp.uint32)
  ct32.polynomial = jnp.where(
      ct32.polynomial >= _moduli, ct32.polynomial - _moduli, ct32.polynomial
  )
  _ct33_arg_data = ct32.polynomial if hasattr(ct32, "polynomial") else ct32
  _ct33_arg_m_in = _ct33_arg_data.shape[-1]
  _ct33_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct33_arg_m_in
  )
  _ct33_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct33_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct33_arg_r)
  )
  _ct33_arg_moduli = getattr(ct32, "moduli", v0.q_towers)
  if isinstance(_ct33_arg_moduli, (int, np.integer)):
    _ct33_arg_moduli = [int(_ct33_arg_moduli)]
  ct33_arg = Polynomial(
      {
          "batch": _ct33_arg_data.shape[0],
          "num_elements": _ct33_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct33_arg_m,
          "precision": 32,
          "degree_layout": (_ct33_arg_r, _ct33_arg_c),
      },
      {"moduli": list(_ct33_arg_moduli)[:_ct33_arg_m]},
  )
  ct33_arg.polynomial = _ct33_arg_data.reshape(
      _ct33_arg_data.shape[0],
      _ct33_arg_data.shape[1],
      _ct33_arg_r,
      _ct33_arg_c,
      _ct33_arg_m_in,
  )[..., :_ct33_arg_m].copy()
  ct33_arg.batch = ct33_arg.polynomial.shape[0]
  ct33_arg.num_elements = ct33_arg.polynomial.shape[1]
  ct33_arg.num_moduli = _ct33_arg_m
  ct33_arg.degree_layout = (_ct33_arg_r, _ct33_arg_c)
  ct33_arg.r = _ct33_arg_r
  ct33_arg.c = _ct33_arg_c
  ct33_arg.moduli = list(_ct33_arg_moduli)[:_ct33_arg_m]
  ct33_arg.moduli_array = jnp.array(
      ct33_arg.moduli, dtype=getattr(ct33_arg, "modulus_dtype", jnp.uint32)
  )
  ct33_raw = v0.he_rot[v0.max_level - 1, 3].rotate(ct33_arg)
  _ct33_data = (
      ct33_raw.polynomial if hasattr(ct33_raw, "polynomial") else ct33_raw
  )
  _ct33_m_in = _ct33_data.shape[-1]
  _ct33_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct33_m_in
  )
  _ct33_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct33_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct33_r)
  )
  _ct33_moduli = getattr(ct33_raw, "moduli", v0.q_towers)
  if isinstance(_ct33_moduli, (int, np.integer)):
    _ct33_moduli = [int(_ct33_moduli)]
  ct33 = Polynomial(
      {
          "batch": _ct33_data.shape[0],
          "num_elements": _ct33_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct33_m,
          "precision": 32,
          "degree_layout": (_ct33_r, _ct33_c),
      },
      {"moduli": list(_ct33_moduli)[:_ct33_m]},
  )
  ct33.polynomial = _ct33_data.reshape(
      _ct33_data.shape[0], _ct33_data.shape[1], _ct33_r, _ct33_c, _ct33_m_in
  )[..., :_ct33_m].copy()
  ct33.batch = ct33.polynomial.shape[0]
  ct33.num_elements = ct33.polynomial.shape[1]
  ct33.num_moduli = _ct33_m
  ct33.degree_layout = (_ct33_r, _ct33_c)
  ct33.r = _ct33_r
  ct33.c = _ct33_c
  ct33.moduli = list(_ct33_moduli)[:_ct33_m]
  ct33.moduli_array = jnp.array(
      ct33.moduli, dtype=getattr(ct33, "modulus_dtype", jnp.uint32)
  )
  _ct34_arg_data = ct20.polynomial if hasattr(ct20, "polynomial") else ct20
  _ct34_arg_m_in = _ct34_arg_data.shape[-1]
  _ct34_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct34_arg_m_in
  )
  _ct34_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct34_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct34_arg_r)
  )
  _ct34_arg_moduli = getattr(ct20, "moduli", v0.q_towers)
  if isinstance(_ct34_arg_moduli, (int, np.integer)):
    _ct34_arg_moduli = [int(_ct34_arg_moduli)]
  ct34_arg = Polynomial(
      {
          "batch": _ct34_arg_data.shape[0],
          "num_elements": _ct34_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct34_arg_m,
          "precision": 32,
          "degree_layout": (_ct34_arg_r, _ct34_arg_c),
      },
      {"moduli": list(_ct34_arg_moduli)[:_ct34_arg_m]},
  )
  ct34_arg.polynomial = _ct34_arg_data.reshape(
      _ct34_arg_data.shape[0],
      _ct34_arg_data.shape[1],
      _ct34_arg_r,
      _ct34_arg_c,
      _ct34_arg_m_in,
  )[..., :_ct34_arg_m].copy()
  ct34_arg.batch = ct34_arg.polynomial.shape[0]
  ct34_arg.num_elements = ct34_arg.polynomial.shape[1]
  ct34_arg.num_moduli = _ct34_arg_m
  ct34_arg.degree_layout = (_ct34_arg_r, _ct34_arg_c)
  ct34_arg.r = _ct34_arg_r
  ct34_arg.c = _ct34_arg_c
  ct34_arg.moduli = list(_ct34_arg_moduli)[:_ct34_arg_m]
  ct34_arg.moduli_array = jnp.array(
      ct34_arg.moduli, dtype=getattr(ct34_arg, "modulus_dtype", jnp.uint32)
  )
  ct34_pt_ntt = (
      pt14.polynomial[0, 0, :, : ct34_arg.polynomial.shape[-1]]
      .reshape(ct34_arg.r, ct34_arg.c, ct34_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct34_ptct = v0.ptct_mul[v0.max_level - 1]
  ct34_ptct.set_plaintext(ct34_pt_ntt)
  ct34_raw = ct34_ptct.mul(ct34_arg, use_bat=False)
  _ct34_data = (
      ct34_raw.polynomial if hasattr(ct34_raw, "polynomial") else ct34_raw
  )
  _ct34_m_in = _ct34_data.shape[-1]
  _ct34_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct34_m_in
  )
  _ct34_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct34_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct34_r)
  )
  _ct34_moduli = getattr(ct34_raw, "moduli", v0.q_towers)
  if isinstance(_ct34_moduli, (int, np.integer)):
    _ct34_moduli = [int(_ct34_moduli)]
  ct34 = Polynomial(
      {
          "batch": _ct34_data.shape[0],
          "num_elements": _ct34_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct34_m,
          "precision": 32,
          "degree_layout": (_ct34_r, _ct34_c),
      },
      {"moduli": list(_ct34_moduli)[:_ct34_m]},
  )
  ct34.polynomial = _ct34_data.reshape(
      _ct34_data.shape[0], _ct34_data.shape[1], _ct34_r, _ct34_c, _ct34_m_in
  )[..., :_ct34_m].copy()
  ct34.batch = ct34.polynomial.shape[0]
  ct34.num_elements = ct34.polynomial.shape[1]
  ct34.num_moduli = _ct34_m
  ct34.degree_layout = (_ct34_r, _ct34_c)
  ct34.r = _ct34_r
  ct34.c = _ct34_c
  ct34.moduli = list(_ct34_moduli)[:_ct34_m]
  ct34.moduli_array = jnp.array(
      ct34.moduli, dtype=getattr(ct34, "modulus_dtype", jnp.uint32)
  )
  _ct35_arg_data = ct23.polynomial if hasattr(ct23, "polynomial") else ct23
  _ct35_arg_m_in = _ct35_arg_data.shape[-1]
  _ct35_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct35_arg_m_in
  )
  _ct35_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct35_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct35_arg_r)
  )
  _ct35_arg_moduli = getattr(ct23, "moduli", v0.q_towers)
  if isinstance(_ct35_arg_moduli, (int, np.integer)):
    _ct35_arg_moduli = [int(_ct35_arg_moduli)]
  ct35_arg = Polynomial(
      {
          "batch": _ct35_arg_data.shape[0],
          "num_elements": _ct35_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct35_arg_m,
          "precision": 32,
          "degree_layout": (_ct35_arg_r, _ct35_arg_c),
      },
      {"moduli": list(_ct35_arg_moduli)[:_ct35_arg_m]},
  )
  ct35_arg.polynomial = _ct35_arg_data.reshape(
      _ct35_arg_data.shape[0],
      _ct35_arg_data.shape[1],
      _ct35_arg_r,
      _ct35_arg_c,
      _ct35_arg_m_in,
  )[..., :_ct35_arg_m].copy()
  ct35_arg.batch = ct35_arg.polynomial.shape[0]
  ct35_arg.num_elements = ct35_arg.polynomial.shape[1]
  ct35_arg.num_moduli = _ct35_arg_m
  ct35_arg.degree_layout = (_ct35_arg_r, _ct35_arg_c)
  ct35_arg.r = _ct35_arg_r
  ct35_arg.c = _ct35_arg_c
  ct35_arg.moduli = list(_ct35_arg_moduli)[:_ct35_arg_m]
  ct35_arg.moduli_array = jnp.array(
      ct35_arg.moduli, dtype=getattr(ct35_arg, "modulus_dtype", jnp.uint32)
  )
  ct35_pt_ntt = (
      pt15.polynomial[0, 0, :, : ct35_arg.polynomial.shape[-1]]
      .reshape(ct35_arg.r, ct35_arg.c, ct35_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct35_ptct = v0.ptct_mul[v0.max_level - 1]
  ct35_ptct.set_plaintext(ct35_pt_ntt)
  ct35_raw = ct35_ptct.mul(ct35_arg, use_bat=False)
  _ct35_data = (
      ct35_raw.polynomial if hasattr(ct35_raw, "polynomial") else ct35_raw
  )
  _ct35_m_in = _ct35_data.shape[-1]
  _ct35_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct35_m_in
  )
  _ct35_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct35_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct35_r)
  )
  _ct35_moduli = getattr(ct35_raw, "moduli", v0.q_towers)
  if isinstance(_ct35_moduli, (int, np.integer)):
    _ct35_moduli = [int(_ct35_moduli)]
  ct35 = Polynomial(
      {
          "batch": _ct35_data.shape[0],
          "num_elements": _ct35_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct35_m,
          "precision": 32,
          "degree_layout": (_ct35_r, _ct35_c),
      },
      {"moduli": list(_ct35_moduli)[:_ct35_m]},
  )
  ct35.polynomial = _ct35_data.reshape(
      _ct35_data.shape[0], _ct35_data.shape[1], _ct35_r, _ct35_c, _ct35_m_in
  )[..., :_ct35_m].copy()
  ct35.batch = ct35.polynomial.shape[0]
  ct35.num_elements = ct35.polynomial.shape[1]
  ct35.num_moduli = _ct35_m
  ct35.degree_layout = (_ct35_r, _ct35_c)
  ct35.r = _ct35_r
  ct35.c = _ct35_c
  ct35.moduli = list(_ct35_moduli)[:_ct35_m]
  ct35.moduli_array = jnp.array(
      ct35.moduli, dtype=getattr(ct35, "modulus_dtype", jnp.uint32)
  )
  _ct36_data = ct34.polynomial if hasattr(ct34, "polynomial") else ct34
  _ct36_m_in = _ct36_data.shape[-1]
  _ct36_m = _ct36_m_in
  _ct36_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct36_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct36_r)
  )
  _ct36_moduli = getattr(ct34, "moduli", v0.q_towers)
  if isinstance(_ct36_moduli, (int, np.integer)):
    _ct36_moduli = [int(_ct36_moduli)]
  ct36 = Polynomial(
      {
          "batch": _ct36_data.shape[0],
          "num_elements": _ct36_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct36_m,
          "precision": 32,
          "degree_layout": (_ct36_r, _ct36_c),
      },
      {"moduli": list(_ct36_moduli)[:_ct36_m]},
  )
  ct36.polynomial = _ct36_data.reshape(
      _ct36_data.shape[0], _ct36_data.shape[1], _ct36_r, _ct36_c, _ct36_m_in
  )[..., :_ct36_m].copy()
  ct36.batch = ct36.polynomial.shape[0]
  ct36.num_elements = ct36.polynomial.shape[1]
  ct36.num_moduli = _ct36_m
  ct36.degree_layout = (_ct36_r, _ct36_c)
  ct36.r = _ct36_r
  ct36.c = _ct36_c
  ct36.moduli = list(_ct36_moduli)[:_ct36_m]
  ct36.moduli_array = jnp.array(
      ct36.moduli, dtype=getattr(ct36, "modulus_dtype", jnp.uint32)
  )
  _ct36_rhs_data = ct35.polynomial if hasattr(ct35, "polynomial") else ct35
  _ct36_rhs_m_in = _ct36_rhs_data.shape[-1]
  _ct36_rhs_m = _ct36_rhs_m_in
  _ct36_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct36_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct36_rhs_r)
  )
  _ct36_rhs_moduli = getattr(ct35, "moduli", v0.q_towers)
  if isinstance(_ct36_rhs_moduli, (int, np.integer)):
    _ct36_rhs_moduli = [int(_ct36_rhs_moduli)]
  ct36_rhs = Polynomial(
      {
          "batch": _ct36_rhs_data.shape[0],
          "num_elements": _ct36_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct36_rhs_m,
          "precision": 32,
          "degree_layout": (_ct36_rhs_r, _ct36_rhs_c),
      },
      {"moduli": list(_ct36_rhs_moduli)[:_ct36_rhs_m]},
  )
  ct36_rhs.polynomial = _ct36_rhs_data.reshape(
      _ct36_rhs_data.shape[0],
      _ct36_rhs_data.shape[1],
      _ct36_rhs_r,
      _ct36_rhs_c,
      _ct36_rhs_m_in,
  )[..., :_ct36_rhs_m].copy()
  ct36_rhs.batch = ct36_rhs.polynomial.shape[0]
  ct36_rhs.num_elements = ct36_rhs.polynomial.shape[1]
  ct36_rhs.num_moduli = _ct36_rhs_m
  ct36_rhs.degree_layout = (_ct36_rhs_r, _ct36_rhs_c)
  ct36_rhs.r = _ct36_rhs_r
  ct36_rhs.c = _ct36_rhs_c
  ct36_rhs.moduli = list(_ct36_rhs_moduli)[:_ct36_rhs_m]
  ct36_rhs.moduli_array = jnp.array(
      ct36_rhs.moduli, dtype=getattr(ct36_rhs, "modulus_dtype", jnp.uint32)
  )
  ct36.add(ct36_rhs)
  _moduli = jnp.array(ct36.moduli, dtype=jnp.uint32)
  ct36.polynomial = jnp.where(
      ct36.polynomial >= _moduli, ct36.polynomial - _moduli, ct36.polynomial
  )
  _ct37_arg_data = ct36.polynomial if hasattr(ct36, "polynomial") else ct36
  _ct37_arg_m_in = _ct37_arg_data.shape[-1]
  _ct37_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct37_arg_m_in
  )
  _ct37_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct37_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct37_arg_r)
  )
  _ct37_arg_moduli = getattr(ct36, "moduli", v0.q_towers)
  if isinstance(_ct37_arg_moduli, (int, np.integer)):
    _ct37_arg_moduli = [int(_ct37_arg_moduli)]
  ct37_arg = Polynomial(
      {
          "batch": _ct37_arg_data.shape[0],
          "num_elements": _ct37_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct37_arg_m,
          "precision": 32,
          "degree_layout": (_ct37_arg_r, _ct37_arg_c),
      },
      {"moduli": list(_ct37_arg_moduli)[:_ct37_arg_m]},
  )
  ct37_arg.polynomial = _ct37_arg_data.reshape(
      _ct37_arg_data.shape[0],
      _ct37_arg_data.shape[1],
      _ct37_arg_r,
      _ct37_arg_c,
      _ct37_arg_m_in,
  )[..., :_ct37_arg_m].copy()
  ct37_arg.batch = ct37_arg.polynomial.shape[0]
  ct37_arg.num_elements = ct37_arg.polynomial.shape[1]
  ct37_arg.num_moduli = _ct37_arg_m
  ct37_arg.degree_layout = (_ct37_arg_r, _ct37_arg_c)
  ct37_arg.r = _ct37_arg_r
  ct37_arg.c = _ct37_arg_c
  ct37_arg.moduli = list(_ct37_arg_moduli)[:_ct37_arg_m]
  ct37_arg.moduli_array = jnp.array(
      ct37_arg.moduli, dtype=getattr(ct37_arg, "modulus_dtype", jnp.uint32)
  )
  ct37_raw = v0.he_rot[v0.max_level - 1, 6].rotate(ct37_arg)
  _ct37_data = (
      ct37_raw.polynomial if hasattr(ct37_raw, "polynomial") else ct37_raw
  )
  _ct37_m_in = _ct37_data.shape[-1]
  _ct37_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct37_m_in
  )
  _ct37_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct37_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct37_r)
  )
  _ct37_moduli = getattr(ct37_raw, "moduli", v0.q_towers)
  if isinstance(_ct37_moduli, (int, np.integer)):
    _ct37_moduli = [int(_ct37_moduli)]
  ct37 = Polynomial(
      {
          "batch": _ct37_data.shape[0],
          "num_elements": _ct37_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct37_m,
          "precision": 32,
          "degree_layout": (_ct37_r, _ct37_c),
      },
      {"moduli": list(_ct37_moduli)[:_ct37_m]},
  )
  ct37.polynomial = _ct37_data.reshape(
      _ct37_data.shape[0], _ct37_data.shape[1], _ct37_r, _ct37_c, _ct37_m_in
  )[..., :_ct37_m].copy()
  ct37.batch = ct37.polynomial.shape[0]
  ct37.num_elements = ct37.polynomial.shape[1]
  ct37.num_moduli = _ct37_m
  ct37.degree_layout = (_ct37_r, _ct37_c)
  ct37.r = _ct37_r
  ct37.c = _ct37_c
  ct37.moduli = list(_ct37_moduli)[:_ct37_m]
  ct37.moduli_array = jnp.array(
      ct37.moduli, dtype=getattr(ct37, "modulus_dtype", jnp.uint32)
  )
  _ct38_data = ct21.polynomial if hasattr(ct21, "polynomial") else ct21
  _ct38_m_in = _ct38_data.shape[-1]
  _ct38_m = _ct38_m_in
  _ct38_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct38_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct38_r)
  )
  _ct38_moduli = getattr(ct21, "moduli", v0.q_towers)
  if isinstance(_ct38_moduli, (int, np.integer)):
    _ct38_moduli = [int(_ct38_moduli)]
  ct38 = Polynomial(
      {
          "batch": _ct38_data.shape[0],
          "num_elements": _ct38_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct38_m,
          "precision": 32,
          "degree_layout": (_ct38_r, _ct38_c),
      },
      {"moduli": list(_ct38_moduli)[:_ct38_m]},
  )
  ct38.polynomial = _ct38_data.reshape(
      _ct38_data.shape[0], _ct38_data.shape[1], _ct38_r, _ct38_c, _ct38_m_in
  )[..., :_ct38_m].copy()
  ct38.batch = ct38.polynomial.shape[0]
  ct38.num_elements = ct38.polynomial.shape[1]
  ct38.num_moduli = _ct38_m
  ct38.degree_layout = (_ct38_r, _ct38_c)
  ct38.r = _ct38_r
  ct38.c = _ct38_c
  ct38.moduli = list(_ct38_moduli)[:_ct38_m]
  ct38.moduli_array = jnp.array(
      ct38.moduli, dtype=getattr(ct38, "modulus_dtype", jnp.uint32)
  )
  _ct38_rhs_data = ct24.polynomial if hasattr(ct24, "polynomial") else ct24
  _ct38_rhs_m_in = _ct38_rhs_data.shape[-1]
  _ct38_rhs_m = _ct38_rhs_m_in
  _ct38_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct38_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct38_rhs_r)
  )
  _ct38_rhs_moduli = getattr(ct24, "moduli", v0.q_towers)
  if isinstance(_ct38_rhs_moduli, (int, np.integer)):
    _ct38_rhs_moduli = [int(_ct38_rhs_moduli)]
  ct38_rhs = Polynomial(
      {
          "batch": _ct38_rhs_data.shape[0],
          "num_elements": _ct38_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct38_rhs_m,
          "precision": 32,
          "degree_layout": (_ct38_rhs_r, _ct38_rhs_c),
      },
      {"moduli": list(_ct38_rhs_moduli)[:_ct38_rhs_m]},
  )
  ct38_rhs.polynomial = _ct38_rhs_data.reshape(
      _ct38_rhs_data.shape[0],
      _ct38_rhs_data.shape[1],
      _ct38_rhs_r,
      _ct38_rhs_c,
      _ct38_rhs_m_in,
  )[..., :_ct38_rhs_m].copy()
  ct38_rhs.batch = ct38_rhs.polynomial.shape[0]
  ct38_rhs.num_elements = ct38_rhs.polynomial.shape[1]
  ct38_rhs.num_moduli = _ct38_rhs_m
  ct38_rhs.degree_layout = (_ct38_rhs_r, _ct38_rhs_c)
  ct38_rhs.r = _ct38_rhs_r
  ct38_rhs.c = _ct38_rhs_c
  ct38_rhs.moduli = list(_ct38_rhs_moduli)[:_ct38_rhs_m]
  ct38_rhs.moduli_array = jnp.array(
      ct38_rhs.moduli, dtype=getattr(ct38_rhs, "modulus_dtype", jnp.uint32)
  )
  ct38.add(ct38_rhs)
  _moduli = jnp.array(ct38.moduli, dtype=jnp.uint32)
  ct38.polynomial = jnp.where(
      ct38.polynomial >= _moduli, ct38.polynomial - _moduli, ct38.polynomial
  )
  _ct39_data = ct27.polynomial if hasattr(ct27, "polynomial") else ct27
  _ct39_m_in = _ct39_data.shape[-1]
  _ct39_m = _ct39_m_in
  _ct39_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct39_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct39_r)
  )
  _ct39_moduli = getattr(ct27, "moduli", v0.q_towers)
  if isinstance(_ct39_moduli, (int, np.integer)):
    _ct39_moduli = [int(_ct39_moduli)]
  ct39 = Polynomial(
      {
          "batch": _ct39_data.shape[0],
          "num_elements": _ct39_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct39_m,
          "precision": 32,
          "degree_layout": (_ct39_r, _ct39_c),
      },
      {"moduli": list(_ct39_moduli)[:_ct39_m]},
  )
  ct39.polynomial = _ct39_data.reshape(
      _ct39_data.shape[0], _ct39_data.shape[1], _ct39_r, _ct39_c, _ct39_m_in
  )[..., :_ct39_m].copy()
  ct39.batch = ct39.polynomial.shape[0]
  ct39.num_elements = ct39.polynomial.shape[1]
  ct39.num_moduli = _ct39_m
  ct39.degree_layout = (_ct39_r, _ct39_c)
  ct39.r = _ct39_r
  ct39.c = _ct39_c
  ct39.moduli = list(_ct39_moduli)[:_ct39_m]
  ct39.moduli_array = jnp.array(
      ct39.moduli, dtype=getattr(ct39, "modulus_dtype", jnp.uint32)
  )
  _ct39_rhs_data = ct33.polynomial if hasattr(ct33, "polynomial") else ct33
  _ct39_rhs_m_in = _ct39_rhs_data.shape[-1]
  _ct39_rhs_m = _ct39_rhs_m_in
  _ct39_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct39_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct39_rhs_r)
  )
  _ct39_rhs_moduli = getattr(ct33, "moduli", v0.q_towers)
  if isinstance(_ct39_rhs_moduli, (int, np.integer)):
    _ct39_rhs_moduli = [int(_ct39_rhs_moduli)]
  ct39_rhs = Polynomial(
      {
          "batch": _ct39_rhs_data.shape[0],
          "num_elements": _ct39_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct39_rhs_m,
          "precision": 32,
          "degree_layout": (_ct39_rhs_r, _ct39_rhs_c),
      },
      {"moduli": list(_ct39_rhs_moduli)[:_ct39_rhs_m]},
  )
  ct39_rhs.polynomial = _ct39_rhs_data.reshape(
      _ct39_rhs_data.shape[0],
      _ct39_rhs_data.shape[1],
      _ct39_rhs_r,
      _ct39_rhs_c,
      _ct39_rhs_m_in,
  )[..., :_ct39_rhs_m].copy()
  ct39_rhs.batch = ct39_rhs.polynomial.shape[0]
  ct39_rhs.num_elements = ct39_rhs.polynomial.shape[1]
  ct39_rhs.num_moduli = _ct39_rhs_m
  ct39_rhs.degree_layout = (_ct39_rhs_r, _ct39_rhs_c)
  ct39_rhs.r = _ct39_rhs_r
  ct39_rhs.c = _ct39_rhs_c
  ct39_rhs.moduli = list(_ct39_rhs_moduli)[:_ct39_rhs_m]
  ct39_rhs.moduli_array = jnp.array(
      ct39_rhs.moduli, dtype=getattr(ct39_rhs, "modulus_dtype", jnp.uint32)
  )
  ct39.add(ct39_rhs)
  _moduli = jnp.array(ct39.moduli, dtype=jnp.uint32)
  ct39.polynomial = jnp.where(
      ct39.polynomial >= _moduli, ct39.polynomial - _moduli, ct39.polynomial
  )
  _ct40_data = ct39.polynomial if hasattr(ct39, "polynomial") else ct39
  _ct40_m_in = _ct40_data.shape[-1]
  _ct40_m = _ct40_m_in
  _ct40_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct40_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct40_r)
  )
  _ct40_moduli = getattr(ct39, "moduli", v0.q_towers)
  if isinstance(_ct40_moduli, (int, np.integer)):
    _ct40_moduli = [int(_ct40_moduli)]
  ct40 = Polynomial(
      {
          "batch": _ct40_data.shape[0],
          "num_elements": _ct40_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct40_m,
          "precision": 32,
          "degree_layout": (_ct40_r, _ct40_c),
      },
      {"moduli": list(_ct40_moduli)[:_ct40_m]},
  )
  ct40.polynomial = _ct40_data.reshape(
      _ct40_data.shape[0], _ct40_data.shape[1], _ct40_r, _ct40_c, _ct40_m_in
  )[..., :_ct40_m].copy()
  ct40.batch = ct40.polynomial.shape[0]
  ct40.num_elements = ct40.polynomial.shape[1]
  ct40.num_moduli = _ct40_m
  ct40.degree_layout = (_ct40_r, _ct40_c)
  ct40.r = _ct40_r
  ct40.c = _ct40_c
  ct40.moduli = list(_ct40_moduli)[:_ct40_m]
  ct40.moduli_array = jnp.array(
      ct40.moduli, dtype=getattr(ct40, "modulus_dtype", jnp.uint32)
  )
  _ct40_rhs_data = ct37.polynomial if hasattr(ct37, "polynomial") else ct37
  _ct40_rhs_m_in = _ct40_rhs_data.shape[-1]
  _ct40_rhs_m = _ct40_rhs_m_in
  _ct40_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct40_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct40_rhs_r)
  )
  _ct40_rhs_moduli = getattr(ct37, "moduli", v0.q_towers)
  if isinstance(_ct40_rhs_moduli, (int, np.integer)):
    _ct40_rhs_moduli = [int(_ct40_rhs_moduli)]
  ct40_rhs = Polynomial(
      {
          "batch": _ct40_rhs_data.shape[0],
          "num_elements": _ct40_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct40_rhs_m,
          "precision": 32,
          "degree_layout": (_ct40_rhs_r, _ct40_rhs_c),
      },
      {"moduli": list(_ct40_rhs_moduli)[:_ct40_rhs_m]},
  )
  ct40_rhs.polynomial = _ct40_rhs_data.reshape(
      _ct40_rhs_data.shape[0],
      _ct40_rhs_data.shape[1],
      _ct40_rhs_r,
      _ct40_rhs_c,
      _ct40_rhs_m_in,
  )[..., :_ct40_rhs_m].copy()
  ct40_rhs.batch = ct40_rhs.polynomial.shape[0]
  ct40_rhs.num_elements = ct40_rhs.polynomial.shape[1]
  ct40_rhs.num_moduli = _ct40_rhs_m
  ct40_rhs.degree_layout = (_ct40_rhs_r, _ct40_rhs_c)
  ct40_rhs.r = _ct40_rhs_r
  ct40_rhs.c = _ct40_rhs_c
  ct40_rhs.moduli = list(_ct40_rhs_moduli)[:_ct40_rhs_m]
  ct40_rhs.moduli_array = jnp.array(
      ct40_rhs.moduli, dtype=getattr(ct40_rhs, "modulus_dtype", jnp.uint32)
  )
  ct40.add(ct40_rhs)
  _moduli = jnp.array(ct40.moduli, dtype=jnp.uint32)
  ct40.polynomial = jnp.where(
      ct40.polynomial >= _moduli, ct40.polynomial - _moduli, ct40.polynomial
  )
  _ct41_data = ct38.polynomial if hasattr(ct38, "polynomial") else ct38
  _ct41_m_in = _ct41_data.shape[-1]
  _ct41_m = _ct41_m_in
  _ct41_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct41_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct41_r)
  )
  _ct41_moduli = getattr(ct38, "moduli", v0.q_towers)
  if isinstance(_ct41_moduli, (int, np.integer)):
    _ct41_moduli = [int(_ct41_moduli)]
  ct41 = Polynomial(
      {
          "batch": _ct41_data.shape[0],
          "num_elements": _ct41_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct41_m,
          "precision": 32,
          "degree_layout": (_ct41_r, _ct41_c),
      },
      {"moduli": list(_ct41_moduli)[:_ct41_m]},
  )
  ct41.polynomial = _ct41_data.reshape(
      _ct41_data.shape[0], _ct41_data.shape[1], _ct41_r, _ct41_c, _ct41_m_in
  )[..., :_ct41_m].copy()
  ct41.batch = ct41.polynomial.shape[0]
  ct41.num_elements = ct41.polynomial.shape[1]
  ct41.num_moduli = _ct41_m
  ct41.degree_layout = (_ct41_r, _ct41_c)
  ct41.r = _ct41_r
  ct41.c = _ct41_c
  ct41.moduli = list(_ct41_moduli)[:_ct41_m]
  ct41.moduli_array = jnp.array(
      ct41.moduli, dtype=getattr(ct41, "modulus_dtype", jnp.uint32)
  )
  _ct41_rhs_data = ct40.polynomial if hasattr(ct40, "polynomial") else ct40
  _ct41_rhs_m_in = _ct41_rhs_data.shape[-1]
  _ct41_rhs_m = _ct41_rhs_m_in
  _ct41_rhs_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct41_rhs_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct41_rhs_r)
  )
  _ct41_rhs_moduli = getattr(ct40, "moduli", v0.q_towers)
  if isinstance(_ct41_rhs_moduli, (int, np.integer)):
    _ct41_rhs_moduli = [int(_ct41_rhs_moduli)]
  ct41_rhs = Polynomial(
      {
          "batch": _ct41_rhs_data.shape[0],
          "num_elements": _ct41_rhs_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct41_rhs_m,
          "precision": 32,
          "degree_layout": (_ct41_rhs_r, _ct41_rhs_c),
      },
      {"moduli": list(_ct41_rhs_moduli)[:_ct41_rhs_m]},
  )
  ct41_rhs.polynomial = _ct41_rhs_data.reshape(
      _ct41_rhs_data.shape[0],
      _ct41_rhs_data.shape[1],
      _ct41_rhs_r,
      _ct41_rhs_c,
      _ct41_rhs_m_in,
  )[..., :_ct41_rhs_m].copy()
  ct41_rhs.batch = ct41_rhs.polynomial.shape[0]
  ct41_rhs.num_elements = ct41_rhs.polynomial.shape[1]
  ct41_rhs.num_moduli = _ct41_rhs_m
  ct41_rhs.degree_layout = (_ct41_rhs_r, _ct41_rhs_c)
  ct41_rhs.r = _ct41_rhs_r
  ct41_rhs.c = _ct41_rhs_c
  ct41_rhs.moduli = list(_ct41_rhs_moduli)[:_ct41_rhs_m]
  ct41_rhs.moduli_array = jnp.array(
      ct41_rhs.moduli, dtype=getattr(ct41_rhs, "modulus_dtype", jnp.uint32)
  )
  ct41.add(ct41_rhs)
  _moduli = jnp.array(ct41.moduli, dtype=jnp.uint32)
  ct41.polynomial = jnp.where(
      ct41.polynomial >= _moduli, ct41.polynomial - _moduli, ct41.polynomial
  )
  v20 = [None] * 1
  _ct42_arg_data = ct41.polynomial if hasattr(ct41, "polynomial") else ct41
  _ct42_arg_m_in = _ct42_arg_data.shape[-1]
  _ct42_arg_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 1)
      if hasattr(v0, "_param_cache")
      else _ct42_arg_m_in
  )
  _ct42_arg_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct42_arg_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct42_arg_r)
  )
  _ct42_arg_moduli = getattr(ct41, "moduli", v0.q_towers)
  if isinstance(_ct42_arg_moduli, (int, np.integer)):
    _ct42_arg_moduli = [int(_ct42_arg_moduli)]
  ct42_arg = Polynomial(
      {
          "batch": _ct42_arg_data.shape[0],
          "num_elements": _ct42_arg_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct42_arg_m,
          "precision": 32,
          "degree_layout": (_ct42_arg_r, _ct42_arg_c),
      },
      {"moduli": list(_ct42_arg_moduli)[:_ct42_arg_m]},
  )
  ct42_arg.polynomial = _ct42_arg_data.reshape(
      _ct42_arg_data.shape[0],
      _ct42_arg_data.shape[1],
      _ct42_arg_r,
      _ct42_arg_c,
      _ct42_arg_m_in,
  )[..., :_ct42_arg_m].copy()
  ct42_arg.batch = ct42_arg.polynomial.shape[0]
  ct42_arg.num_elements = ct42_arg.polynomial.shape[1]
  ct42_arg.num_moduli = _ct42_arg_m
  ct42_arg.degree_layout = (_ct42_arg_r, _ct42_arg_c)
  ct42_arg.r = _ct42_arg_r
  ct42_arg.c = _ct42_arg_c
  ct42_arg.moduli = list(_ct42_arg_moduli)[:_ct42_arg_m]
  ct42_arg.moduli_array = jnp.array(
      ct42_arg.moduli, dtype=getattr(ct42_arg, "modulus_dtype", jnp.uint32)
  )
  ct42_raw = v0.he_rescale[v0.max_level - 1, v0.max_level - 2](ct42_arg)
  _ct42_data = (
      ct42_raw.polynomial if hasattr(ct42_raw, "polynomial") else ct42_raw
  )
  _ct42_m_in = _ct42_data.shape[-1]
  _ct42_m = (
      v0._param_cache.num_q_at_level(v0.max_level - 2)
      if hasattr(v0, "_param_cache")
      else _ct42_m_in
  )
  _ct42_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct42_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct42_r)
  )
  _ct42_moduli = getattr(ct42_raw, "moduli", v0.q_towers)
  if isinstance(_ct42_moduli, (int, np.integer)):
    _ct42_moduli = [int(_ct42_moduli)]
  ct42 = Polynomial(
      {
          "batch": _ct42_data.shape[0],
          "num_elements": _ct42_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct42_m,
          "precision": 32,
          "degree_layout": (_ct42_r, _ct42_c),
      },
      {"moduli": list(_ct42_moduli)[:_ct42_m]},
  )
  ct42.polynomial = _ct42_data.reshape(
      _ct42_data.shape[0], _ct42_data.shape[1], _ct42_r, _ct42_c, _ct42_m_in
  )[..., :_ct42_m].copy()
  ct42.batch = ct42.polynomial.shape[0]
  ct42.num_elements = ct42.polynomial.shape[1]
  ct42.num_moduli = _ct42_m
  ct42.degree_layout = (_ct42_r, _ct42_c)
  ct42.r = _ct42_r
  ct42.c = _ct42_c
  ct42.moduli = list(_ct42_moduli)[:_ct42_m]
  ct42.moduli_array = jnp.array(
      ct42.moduli, dtype=getattr(ct42, "modulus_dtype", jnp.uint32)
  )
  v20[0] = ct42
  v21 = v20
  return v21


def matvec_chain(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  (v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14) = (
      matvec_chain__preprocessing(v0, v1)
  )
  v15 = matvec_chain__preprocessed(
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14
  )
  return v15


def matvec_chain__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v15)
  v0.public_key = v3
  ct_raw = v0.encrypt(pt)
  _ct_data = ct_raw.polynomial if hasattr(ct_raw, "polynomial") else ct_raw
  _ct_m_in = _ct_data.shape[-1]
  _ct_m = _ct_m_in
  _ct_r = (
      v0._param_cache.r
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("r", int(np.sqrt(v0.degree)))
  )
  _ct_c = (
      v0._param_cache.c
      if hasattr(v0, "_param_cache")
      else v0.parameters.get("c", v0.degree // _ct_r)
  )
  _ct_moduli = getattr(ct_raw, "moduli", v0.q_towers)
  if isinstance(_ct_moduli, (int, np.integer)):
    _ct_moduli = [int(_ct_moduli)]
  ct = Polynomial(
      {
          "batch": _ct_data.shape[0],
          "num_elements": _ct_data.shape[1],
          "degree": v0.degree,
          "num_moduli": _ct_m,
          "precision": 32,
          "degree_layout": (_ct_r, _ct_c),
      },
      {"moduli": list(_ct_moduli)[:_ct_m]},
  )
  ct.polynomial = _ct_data.reshape(
      _ct_data.shape[0], _ct_data.shape[1], _ct_r, _ct_c, _ct_m_in
  )[..., :_ct_m].copy()
  ct.batch = ct.polynomial.shape[0]
  ct.num_elements = ct.polynomial.shape[1]
  ct.num_moduli = _ct_m
  ct.degree_layout = (_ct_r, _ct_c)
  ct.r = _ct_r
  ct.c = _ct_c
  ct.moduli = list(_ct_moduli)[:_ct_m]
  ct.moduli_array = jnp.array(
      ct.moduli, dtype=getattr(ct, "modulus_dtype", jnp.uint32)
  )
  v16 = [ct]
  return v16


def matvec_chain__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 0
  v8 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  _num_moduli = ct.polynomial.shape[-1]
  _q_sub = list(getattr(ct, "moduli", v0.q_towers))[:_num_moduli]
  _ct_for_dec = Polynomial(
      {
          "batch": ct.polynomial.shape[0],
          "num_elements": ct.polynomial.shape[1],
          "degree": v0.degree,
          "precision": 32,
          "num_moduli": _num_moduli,
          "degree_layout": (v0.degree,),
      },
      {"moduli": _q_sub},
  )
  _ct_for_dec.set_batch_polynomial(
      ct.polynomial.reshape(
          ct.polynomial.shape[0], ct.polynomial.shape[1], v0.degree, _num_moduli
      )
  )
  pt = v0.decrypt(_ct_for_dec)
  v9 = v0.decode(pt, is_ntt=False).real.reshape(1, 8)
  v10 = v8.copy()
  for v11 in range(0, 8):
    v13 = int(v11)
    v14 = v9[0, v13]
    v10[v13] = v14
  return v10


def matvec_identity__generate_crypto_context(
    public_key,
    secret_key,
    evaluation_key,
) -> ckks.CKKSContext:
  params = {
      "degree": 16,
      "num_slots": 8,
      "batch": 1,
      "r": 4,
      "c": 4,
      "dnum": 3,
      "numEvalMult": 1,
      "scaling_factor": 563019763943521,
      "q_towers": [1073742881, 1073742721, 1073741441, 1073741857, 524353],
      "p_towers": [1073740609, 1073739937, 1073739649],
      "composite_degree": 1,
      "p": 30,
      "max_bits_in_word": 61,
      "max_bits_value": 9223372036854775295,
      "noise_scale_degree": 1,
      "CKKS_M_FACTOR": 1,
      "public_key": public_key,
      "secret_key": secret_key,
      "evaluation_key": evaluation_key,
  }
  v0 = ckks.CKKSContext(params)
  return v0


def matvec_identity__configure_crypto_context(
    v0: ckks.CKKSContext,
):
  v0.program_initialization(
      total_hemul_levels=v0.max_level,
      total_rotation_indices=[1, 2, 3, 6],
      dnum=3,
      r=4,
      c=4,
      batch=1,
  )
