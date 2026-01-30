#include "lib/Backend/cornami/Dialect/SCIFRBool/Analysis/PerfCounter.h"

namespace mlir {
namespace cornami {
SWCounter& SWCounter::operator+=(const SWCounter& that) {
  this->mult += that.mult;
  this->add += that.add;
  this->ntt += that.ntt;

  return *this;
}

PerfCounter& PerfCounter::operator+=(const PerfCounter& that) {
  this->arch_emu += that.arch_emu;
  this->arch_asic += that.arch_asic;
  this->sw += that.sw;
  return *this;
}

}  // namespace cornami
}  // namespace mlir
