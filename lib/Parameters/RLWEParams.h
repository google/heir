#ifndef LIB_PARAMETERS_RLWEPARAMS_H_
#define LIB_PARAMETERS_RLWEPARAMS_H_

#include <cstdint>
#include <vector>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Parameter for BGV scheme at ModuleOp level
class RLWESchemeParam {
 public:
  RLWESchemeParam(int ringDim, int level, const std::vector<double>& logqi,
                  int dnum, const std::vector<double>& logpi, bool usePublicKey,
                  bool encryptionTechniqueExtended)
      : ringDim(ringDim),
        level(level),
        logqi(logqi),
        dnum(dnum),
        logpi(logpi),
        usePublicKey(usePublicKey),
        encryptionTechniqueExtended(encryptionTechniqueExtended) {}

  RLWESchemeParam(int ringDim, int level, const std::vector<double>& logqi,
                  const std::vector<int64_t>& qi, int dnum,
                  const std::vector<double>& logpi,
                  const std::vector<int64_t>& pi, bool usePublicKey,
                  bool encryptionTechniqueExtended)
      : ringDim(ringDim),
        level(level),
        logqi(logqi),
        qi(qi),
        dnum(dnum),
        logpi(logpi),
        pi(pi),
        usePublicKey(usePublicKey),
        encryptionTechniqueExtended(encryptionTechniqueExtended) {}

  virtual ~RLWESchemeParam() = default;

 protected:
  // the N in Z[X]/(X^N+1)
  int ringDim;

  // the standard deviation of the error distribution
  double std0 = 3.2;

  // RNS level, from 0 to L
  int level;

  // logarithm of the modulus of each level
  // logqi.size() == level + 1
  std::vector<double> logqi;
  // modulus of each level
  std::vector<int64_t> qi;

  // The following part is for HYBRID key switching technique

  // number of digits
  // In HYBRID, we decompose Q into `dnum` digits
  // for example, when Q consists of q0, q1, q2, q3 and dnum = 2,
  // we have two digits: q0q1 and q2q3
  int dnum;
  // logarithm of the special modulus
  std::vector<double> logpi;
  // special modulus
  std::vector<int64_t> pi;

  // whether to use public key
  bool usePublicKey;

  // the encryption technique used.
  // if true, use extended encryption technique.
  // which means encrypt at Qp then mod reduce to Q.
  // this has the benefit of smaller encryption noise.
  bool encryptionTechniqueExtended;

 public:
  int getRingDim() const { return ringDim; }
  int getLevel() const { return level; }
  const std::vector<double>& getLogqi() const { return logqi; }
  const std::vector<int64_t>& getQi() const { return qi; }
  int getDnum() const { return dnum; }
  const std::vector<double>& getLogpi() const { return logpi; }
  const std::vector<int64_t>& getPi() const { return pi; }
  double getStd0() const { return std0; }
  bool getUsePublicKey() const { return usePublicKey; }
  bool isEncryptionTechniqueExtended() const {
    return encryptionTechniqueExtended;
  }

  int64_t getNttPrimitiveRoot() const {
    // FIXME: implement
    return 7;
  }

  virtual void print(llvm::raw_ostream& os) const;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const RLWESchemeParam& param) {
    param.print(os);
    return os;
  }

  static RLWESchemeParam getConservativeRLWESchemeParam(
      int level, int minRingDim, bool usePublicKey,
      bool encryptionTechniqueExtended);

  // plaintext modulus for BGV
  // for CKKS this field is not used
  static RLWESchemeParam getConcreteRLWESchemeParam(
      std::vector<double> logqi, int minRingDim, bool usePublicKey,
      bool encryptionTechniqueExtended, int64_t plaintextModulus = 0);
};

// Parameter for each RLWE ciphertext SSA value.
class RLWELocalParam {
 public:
  RLWELocalParam(const RLWESchemeParam* schemeParam, int currentLevel,
                 int dimension)
      : schemeParam(schemeParam),
        currentLevel(currentLevel),
        dimension(dimension) {}

 protected:
  const RLWESchemeParam* schemeParam;
  int currentLevel;
  int dimension;

 public:
  const RLWESchemeParam* getRLWESchemeParam() const { return schemeParam; }

  int getCurrentLevel() const { return currentLevel; }
  int getDimension() const { return dimension; }
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_RLWEPARAMS_H_
