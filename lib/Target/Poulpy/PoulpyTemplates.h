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
use poulpy_ckks::{
    CKKSMeta, SetCKKSInfos,
    api::{CKKSAddOps, CKKSSubOps, CKKSMulOps, CKKSRotateOps, CKKSCopyOps,
          CKKSPow2Ops, CKKSEncrypt, CKKSDecrypt},
    encoding::Encoder,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext,
              UnnormalizedCKKSCiphertext},
};
use poulpy_core::layouts::prepared::{GLWESecretPrepared, GLWETensorKeyPrepared};
use poulpy_core::layouts::{Base2K, GetDegree, GLWEAutomorphismKeyPrepared,
                           GLWEInfos, GLWELayout, LWEInfos, TorusPrecision};
use poulpy_core::EncryptionLayout;
use poulpy_cpu_ref::{FFT64Ref, FFT64ReimTable, NTT4x30Ref};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};)poulpy";

constexpr std::string_view kTypeAliases = R"poulpy(
type Ct = CKKSCiphertext<<BE as Backend>::OwnedBuf>;
type CtUnnorm = UnnormalizedCKKSCiphertext<<BE as Backend>::OwnedBuf>;
type Pt = CKKSPlaintext<<BE as Backend>::OwnedBuf>;
type Sk = GLWESecretPrepared<<BE as Backend>::OwnedBuf, BE>;
type Tsk = GLWETensorKeyPrepared<<BE as Backend>::OwnedBuf, BE>;
type Akm = HashMap<i64, GLWEAutomorphismKeyPrepared<<BE as Backend>::OwnedBuf, BE>>;
)poulpy";
}  // namespace poulpy
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_POULPY_POULPYTEMPLATES_H_
