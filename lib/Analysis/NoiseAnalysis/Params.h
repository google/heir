#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_PARAMS_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_PARAMS_H_

#include <cmath>
#include <vector>

namespace mlir {
namespace heir {

class SchemeParam {
 public:
  SchemeParam(int ringDim, int64_t plaintextModulus, int level,
              const std::vector<int> &logqi, int dnum,
              const std::vector<int> &logpi)
      : ringDim(ringDim),
        plaintextModulus(plaintextModulus),
        level(level),
        logqi(logqi),
        dnum(dnum),
        logpi(logpi) {}

 private:
  // the N in Z[X]/(X^N+1)
  int ringDim;

  // the plaintext modulud for BGV
  int64_t plaintextModulus;

  // RNS level, from 0 to L
  int level;

  // logarithm of the modulus of each level
  // logqi.size() == level + 1
  std::vector<int> logqi;

  // number of digits
  int dnum;
  // logarithm of the special modulus
  // used during key switching
  std::vector<int> logpi;

  // the standard deviation of the error distribution
  double std0 = 3.2;

 public:
  int getRingDim() const { return ringDim; }
  int64_t getPlaintextModulus() const { return plaintextModulus; }
  int getLevel() const { return level; }
  const std::vector<int> &getLogqi() const { return logqi; }
  int getDnum() const { return dnum; }
  const std::vector<int> &getLogpi() const { return logpi; }
  double getStd0() const { return std0; }

  static SchemeParam getConservativeSchemeParam(int level,
                                                int64_t plaintextModulus);
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
};

}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_PARAMS_H_
