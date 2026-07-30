#ifndef LIB_TARGET_POULPY_POULPYTEMPLATES_H_
#define LIB_TARGET_POULPY_POULPYTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace poulpy {
// TODO(mmoro): copied from poulpy/poulpy-cpu-ref/examples/ckks_poly2.rs
constexpr std::string_view kModulePrelude =
    R"poulpy(use anyhow::Result;
use poulpy_ckks::{
    CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos,
    encoding::Encoder,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext},
    leveled::api::{CKKSAllOpsTmpBytes, CKKSDecrypt, CKKSEncrypt, PolynomialEvaluation},
    polynomial::{BSGSPolynomial, Basis, EncodeBSGS, Polynomial},
    power_basis::{PowerBasis, PowerBasisGen},
};
use poulpy_core::{
    EncryptionLayout, GLWETensorKeyEncryptSk,
    layouts::{
        Base2K, Degree, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, LWEInfos, ModuleCoreAlloc, Rank,
        TorusPrecision,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory, GLWETensorKeyPrepared},
    },
};
use poulpy_cpu_ref::{FFT64ReimTable, NTT4x30Ref};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, ScratchOwned},
    source::Source,
};)poulpy";

constexpr std::string_view kTypeAliases = R"poulpy(
type Ct = CKKSCiphertext<<BE as Backend>::OwnedBuf>;
type Tsk = GLWETensorKeyPrepared<<BE as Backend>::OwnedBuf, BE>;
)poulpy";
}  // namespace poulpy
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_POULPY_POULPYTEMPLATES_H_
