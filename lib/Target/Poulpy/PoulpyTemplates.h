#ifndef LIB_TARGET_POULPY_POULPYTEMPLATES_H_
#define LIB_TARGET_POULPY_POULPYTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace poulpy {
constexpr std::string_view kModulePrelude =
    R"poulpy(#![allow(unused_imports, unused_variables, unused_mut, non_snake_case)]

use anyhow::Result;
use std::collections::HashMap;
use poulpy_ckks::prelude::*;
use poulpy_ckks::layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned};
use poulpy_ckks::oep::{CKKSEncodingImpl, CKKSImpl};
use poulpy_core::layouts::prepared::{GLWESecretPrepared, GLWETensorKeyPrepared};
use poulpy_core::layouts::{Base2K, GLWEAutomorphismKeyPrepared, GLWEInfos,
                           LWEInfos, TorusPrecision};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    oep::HalModuleImpl,
    source::Source,
};

/// The backend capabilities the generated code relies on. The caller picks
/// the backend, e.g. `f::<NTT4x30Portable>(...)`.
pub trait PoulpyBackend:
    CKKSImpl + CKKSEncodingImpl<f64> + HalModuleImpl + Backend<OwnedBuf: Clone> {}
impl<T: CKKSImpl + CKKSEncodingImpl<f64> + HalModuleImpl + Backend<OwnedBuf: Clone>>
    PoulpyBackend for T {}
)poulpy";

constexpr std::string_view kTypeAliases = R"poulpy(
type Ct<BE> = CKKSCiphertextOwned<BE>;
type Pt<BE> = CKKSPlaintextOwned<BE>;
type Sk<BE> = GLWESecretPrepared<<BE as Backend>::OwnedBuf, BE>;
type Tsk<BE> = GLWETensorKeyPrepared<<BE as Backend>::OwnedBuf, BE>;
type Akm<BE> = HashMap<i64, GLWEAutomorphismKeyPrepared<<BE as Backend>::OwnedBuf, BE>>;
)poulpy";
}  // namespace poulpy
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_POULPY_POULPYTEMPLATES_H_
