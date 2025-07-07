#ifndef LIB_UTILS_LOGARITHMETIC_H_
#define LIB_UTILS_LOGARITHMETIC_H_

#include <cmath>
#include <limits>

namespace mlir {
namespace heir {

class Log2Arithmetic {
 public:
  /// User-facing constructor that creates a Log2Arithmetic instance
  /// from a double value.
  static Log2Arithmetic of(double value) {
    double log2value;
    if (value == 0.0) {
      log2value = NEGATIVE_INFINITY;
    } else {
      log2value = std::log2(value);
    }
    return Log2Arithmetic(log2value);
  }

  Log2Arithmetic() : log2value(NEGATIVE_INFINITY) {}

  Log2Arithmetic operator+(const Log2Arithmetic &rhs) const;

  Log2Arithmetic operator*(const Log2Arithmetic &rhs) const;

  bool operator==(const Log2Arithmetic &rhs) const {
    return log2value == rhs.log2value;
  }

  bool operator<(const Log2Arithmetic &rhs) const {
    if (log2value == NEGATIVE_INFINITY) {
      return rhs.log2value != NEGATIVE_INFINITY;
    }
    if (rhs.log2value == NEGATIVE_INFINITY) {
      return false;  // this.log2value > NEGATIVE_INFINITY
    }
    return log2value < rhs.log2value;
  }

  /// Returns the storage value of this Log2Arithmetic instance.
  double getLog2Value() const { return log2value; }

  /// Caution: This may overflow if the value is too large.
  double getValue() const {
    if (log2value == NEGATIVE_INFINITY) {
      return 0.0;
    }
    return std::exp2(log2value);
  }

 private:
  /// Internal constructor that initializes the Log2Arithmetic instance
  /// with a precomputed log2 value.
  /// This is used internally to prevent user mixing up the semantics of
  /// Log2Arithmetic and double.
  Log2Arithmetic(double log2value) : log2value(log2value) {}

  // value may be negative infinity
  static constexpr double NEGATIVE_INFINITY =
      -std::numeric_limits<double>::infinity();
  double log2value;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LOGARITHMETIC_H_
