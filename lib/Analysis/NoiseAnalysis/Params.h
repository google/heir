#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_PARAMS_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_PARAMS_H_

#include <cmath>

#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"       // from @llvm-project

namespace mlir {
namespace heir {

// struct RLWEParam {
//   int ringDim;
//   int maxQ;
// };
//
// // uniform tenary
// static struct RLWEParam HEStd_128_classic[] = {
//     {1024, 27},   {2048, 54},   {4096, 109},   {8192, 218},
//     {16384, 438}, {32768, 881}, {65536, 1747}, {131072, 3523}};

class SchemeParam {
 public:
  SchemeParam(int ringDim, int64_t plaintextModulus, int level,
              const std::vector<int> &qi, int dnum, const std::vector<int> &pi)
      : ringDim(ringDim),
        plaintextModulus(plaintextModulus),
        level(level),
        qi(qi),
        dnum(dnum),
        pi(pi) {}

 private:
  // the N in Z[X]/(X^N+1)
  int ringDim;

  // the plaintext modulud for BGV
  int64_t plaintextModulus;

  // RNS level, from 0 to L
  int level;

  // logarithm of the modulus of each level
  // qi.size() == level + 1
  std::vector<int> qi;

  // number of digits
  int dnum;
  // logarithm of the special modulus
  // used during key switching
  std::vector<int> pi;

  // the standard deviation of the error distribution
  double std0 = 3.2;

 public:
  // void print(llvm::raw_ostream &os) const;

  // friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
  //                                      const SchemeParam &param) {
  //   param.print(os);
  //   return os;
  // }

  int getRingDim() const { return ringDim; }
  int64_t getPlaintextModulus() const { return plaintextModulus; }
  int getLevel() const { return level; }
  const std::vector<int> &getQi() const { return qi; }
  int getDnum() const { return dnum; }
  const std::vector<int> &getPi() const { return pi; }
  double getStd0() const { return std0; }
};

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

  // void print(llvm::raw_ostream &os) const;

  // friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
  //                                      const LocalParam &param) {
  //   param.print(os);
  //   return os;
  // }
};

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_PARAMS_H_
