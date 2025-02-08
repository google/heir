#ifndef LIB_PARAMETERS_BGV_PARAMS_H_
#define LIB_PARAMETERS_BGV_PARAMS_H_

#include <cstdint>
#include <vector>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace bgv {

// Parameter for BGV scheme at ModuleOp level
class SchemeParam {
 public:
  SchemeParam(int ringDim, int64_t plaintextModulus, int level,
              const std::vector<double> &logqi, int dnum,
              const std::vector<double> &logpi)
      : ringDim(ringDim),
        plaintextModulus(plaintextModulus),
        level(level),
        logqi(logqi),
        dnum(dnum),
        logpi(logpi) {}

  SchemeParam(int ringDim, int64_t plaintextModulus, int level,
              const std::vector<double> &logqi, const std::vector<int64_t> &qi,
              int dnum, const std::vector<double> &logpi,
              const std::vector<int64_t> &pi)
      : ringDim(ringDim),
        plaintextModulus(plaintextModulus),
        level(level),
        logqi(logqi),
        qi(qi),
        dnum(dnum),
        logpi(logpi),
        pi(pi) {}

 private:
  // the N in Z[X]/(X^N+1)
  int ringDim;

  // the plaintext modulus for BGV
  int64_t plaintextModulus;

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

 public:
  int getRingDim() const { return ringDim; }
  int64_t getPlaintextModulus() const { return plaintextModulus; }
  int getLevel() const { return level; }
  const std::vector<double> &getLogqi() const { return logqi; }
  const std::vector<int64_t> &getQi() const { return qi; }
  int getDnum() const { return dnum; }
  const std::vector<double> &getLogpi() const { return logpi; }
  const std::vector<int64_t> &getPi() const { return pi; }
  double getStd0() const { return std0; }

  void print(llvm::raw_ostream &os) const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const SchemeParam &param) {
    param.print(os);
    return os;
  }

  static SchemeParam getConservativeSchemeParam(int level,
                                                int64_t plaintextModulus);

  static SchemeParam getConcreteSchemeParam(int64_t plaintextModulus,
                                            std::vector<double> logqi);
};

// Parameter for each BGV ciphertext SSA value.
class LocalParam {
 public:
  LocalParam(const SchemeParam *schemeParam, int currentLevel, int dimension)
      : schemeParam(schemeParam),
        currentLevel(currentLevel),
        dimension(dimension) {}

 private:
  const SchemeParam *schemeParam;
  int currentLevel;
  int dimension;

 public:
  const SchemeParam *getSchemeParam() const { return schemeParam; }

  int getCurrentLevel() const { return currentLevel; }
  int getDimension() const { return dimension; }
};

}  // namespace bgv
}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_BGV_PARAMS_H_
