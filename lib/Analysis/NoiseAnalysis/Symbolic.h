#ifndef INCLUDE_ANALYSIS_NOISEANALYSIS_SYMBOLIC_H_
#define INCLUDE_ANALYSIS_NOISEANALYSIS_SYMBOLIC_H_

#include <cassert>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"         // from @llvm-project

namespace mlir {
namespace heir {
namespace noise {

// User should ensure "name" is unique
class Symbol {
 public:
  enum SymbolKind {
    CONSTANT = 0,  // param is constant
    RANDOMVARIABLE,
    /*RV*/ GAUSSIAN,  // param is stderr
    /*RV*/ UNIFORM,
    /*RV*/ UNIFORM_TERNARY,  // [-param/2, param/2]
    END_OF_RV,
  };
  SymbolKind getKind() const { return kind; }
  bool isConstant() const { return kind == CONSTANT; }
  bool isRandomVariable() const {
    return kind >= RANDOMVARIABLE && kind < END_OF_RV;
  }
  double getConstant() const {
    assert(isConstant());
    return param;
  }
  double getVariance() const;

 private:
  const SymbolKind kind;

 public:
  Symbol(SymbolKind kind, const std::string &name, double param)
      : kind(kind), name(name), param(param) {}

  bool operator<(const Symbol &rhs) const { return name < rhs.name; }
  bool operator==(const Symbol &rhs) const { return name == rhs.name; }

  std::string getName() const { return name; }
  double getParam() const { return param; }

 private:
  std::string name;
  // interpreted differently for different symbol
  double param;
};

class Monomial {
 public:
  using ExponentType = int64_t;
  using SymbolsType = std::map<Symbol, ExponentType>;

  Monomial() = default;
  Monomial(const Symbol &symbol) { symbols[symbol] = 1; }

  Monomial(const Symbol &symbol, ExponentType exponent)
      : symbols({{symbol, exponent}}) {}

  Monomial(const SymbolsType &symbols) : symbols(symbols) {}

  bool operator<(const Monomial &rhs) const { return symbols < rhs.symbols; }
  bool operator==(const Monomial &rhs) const { return symbols == rhs.symbols; }

  const SymbolsType &getSymbols() const { return symbols; }

  Monomial operator*(const Monomial &rhs) const;

  std::string toString() const;

 private:
  SymbolsType symbols;
};

class Expression {
 public:
  using CoefficientType = int64_t;
  using MonomialsType = std::map<Monomial, CoefficientType>;

  Expression() = default;
  Expression(const Symbol &symbol) { monomials.insert({Monomial(symbol), 1}); }
  Expression(const Monomial &monomial) { monomials.insert({monomial, 1}); }

  Expression(const MonomialsType &monomials) : monomials(monomials) {}

  const MonomialsType &getMonomials() const { return monomials; }

  bool operator==(const Expression &rhs) const {
    return monomials == rhs.monomials;
  }

  Expression operator+(const Expression &rhs) const;

  Expression operator*(const Expression &rhs) const;

  std::string toString() const;

  void print(raw_ostream &os) const { os << toString(); }

  // for Lattice in DataFlowFramework
  static Expression join(const Expression &lhs, const Expression &rhs) {
    // prefer lhs when equal
    if (lhs.monomials.size() >= rhs.monomials.size()) {
      return lhs;
    }
    return rhs;
  }

  // for compatibility with other Noise Lattice
  bool isInitialized() const { return !monomials.empty(); }

  // random variable related
  double getVariance(int ringDim) const;

 private:
  MonomialsType monomials;
};

using Expr = Expression;

}  // namespace noise
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_ANALYSIS_NOISEANALYSIS_SYMBOLIC_H_
