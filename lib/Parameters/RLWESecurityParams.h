#ifndef LIB_PARAMETERS_RLWESECURITYPARAMS_H_
#define LIB_PARAMETERS_RLWESECURITYPARAMS_H_

namespace mlir {
namespace heir {

// struct for recording the maximal Q for each ring dim
// under certain security condition.
struct RLWESecurityParam {
  int ringDim;
  int logMaxQ;
};

// compute ringDim given logPQ under 128-bit classic security
int computeRingDim(int logPQ, int minRingDim);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_RLWESECURITYPARAMS_H_
