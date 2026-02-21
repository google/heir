#include "lib/Analysis/NoiseAnalysis/Symbolic.h"

#include <cmath>

#include "lib/Analysis/NoiseAnalysis/Noise.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project

#define DEBUG_TYPE "Symbolic"

namespace mlir {
namespace heir {
namespace noise {

//===------------------------------------------------------------------------===//
// Monomial
//===------------------------------------------------------------------------===//

Monomial Monomial::operator*(const Monomial &rhs) const {
  SymbolsType newSymbols;
  auto updateSymbols = [&](const SymbolsType &symbols) {
    for (auto &[symbol, exponent] : symbols) {
      auto find = newSymbols.find(symbol);
      if (find != newSymbols.end()) {
        find->second += exponent;
      } else {
        newSymbols[symbol] = exponent;
      }
    }
  };
  updateSymbols(this->getSymbols());
  updateSymbols(rhs.getSymbols());
  return newSymbols;
}

std::string Monomial::toString() const {
  std::string ret;
  bool firstTime = true;
  for (auto &[symbol, exponent] : symbols) {
    if (!firstTime) {
      ret += " * ";
    }
    ret += symbol.getName();
    if (exponent != 1) {
      ret += "^" + std::to_string(exponent);
    }
    firstTime = false;
  }
  return ret;
}

//===------------------------------------------------------------------------===//
// Expression
//===------------------------------------------------------------------------===//

Expression Expression::operator+(const Expression &rhs) const {
  MonomialsType newMonomials;
  auto updateMonomials = [&](const MonomialsType &monomials) {
    for (auto &[monomial, coefficient] : monomials) {
      auto find = newMonomials.find(monomial);
      if (find != newMonomials.end()) {
        find->second += coefficient;
      } else {
        newMonomials[monomial] = coefficient;
      }
    }
  };
  updateMonomials(this->getMonomials());
  updateMonomials(rhs.getMonomials());
  return newMonomials;
}

Expression Expression::operator*(const Expression &rhs) const {
  MonomialsType newMonomials;
  for (auto &[lhsMonomial, lhsCoefficient] : this->monomials) {
    for (auto &[rhsMonomial, rhsCoefficient] : rhs.monomials) {
      auto newMonomial = lhsMonomial * rhsMonomial;
      auto newCoefficient = lhsCoefficient * rhsCoefficient;
      auto find = newMonomials.find(newMonomial);
      if (find != newMonomials.end()) {
        find->second += newCoefficient;
      } else {
        newMonomials[newMonomial] = newCoefficient;
      }
    }
  }
  return Expression(newMonomials);
}

std::string Expression::toString() const {
  std::string ret;
  bool firstTime = true;
  for (auto &[monomial, coefficient] : monomials) {
    if (!firstTime) {
      ret += " + ";
    }
    if (coefficient != 1.0) {
      ret += std::to_string(coefficient) + " * ";
    }
    ret += monomial.toString();
    firstTime = false;
  }
  return ret;
}

//===------------------------------------------------------------------------===//
// Constant and Random Variable
//===------------------------------------------------------------------------===//

double Symbol::getVariance() const {
  assert(isRandomVariable());
  switch (kind) {
    case SymbolKind::GAUSSIAN:
      return param * param;
    case SymbolKind::UNIFORM_TERNARY:
      return 2.0 / 3.0;
    case SymbolKind::UNIFORM:
      return param * param / 12.;
    default:
      llvm_unreachable("Unknown symbol kind");
  }
}

double Expression::getVariance(int ringDim) const {
  // Group monomials by sharing the same random variables with same order
  // e.g. c1 * s^i * e^j and c2 * s^i * e^j
  // because they should have been merged as (c1 + c2) * s^i * e^j.
  // Technically, this is done by filling all Constant Symbols.
  // We use NoiseState to do log-arithmetic.
  std::map<Monomial, NoiseState> randomVariableToCoefficient;
  for (auto &[monomial, coefficient] : getMonomials()) {
    Monomial randomVariables;
    auto instantiatedCoefficient = NoiseState::of(coefficient);
    for (auto &[symbol, exponent] : monomial.getSymbols()) {
      if (symbol.isConstant()) {
        // multiply constant for exponent times
        for (auto i = 0; i != exponent; ++i) {
          // this happens using log-arithmetic
          instantiatedCoefficient *= symbol.getConstant();
        }
      } else {
        if (randomVariables.getSymbols().empty()) {
          // first time
          randomVariables = Monomial(symbol, exponent);
        } else {
          randomVariables = randomVariables * Monomial(symbol, exponent);
        }
      }
    }
    // update the map
    auto find = randomVariableToCoefficient.find(randomVariables);
    if (find != randomVariableToCoefficient.end()) {
      find->second = find->second + instantiatedCoefficient;
    } else {
      randomVariableToCoefficient[randomVariables] = instantiatedCoefficient;
    }
  }

  LLVM_DEBUG({
    for (auto &[monomial, coefficient] : randomVariableToCoefficient) {
      llvm::dbgs() << "Monomial: " << monomial.toString()
                   << " Coeff: " << coefficient << "\n";
    }
  });

  auto variance = NoiseState::of(0.0);

  // then we can use the map to calculate variance
  for (auto &[monomial, coefficient] : randomVariableToCoefficient) {
    // in Var(), c -> c^2
    auto currentVariance = coefficient * coefficient;

    // calculate s^j e^k using the following formula
    // Var(s^j e^k) = j! * k! * N^{j+k-1} Var(s)^j * Var(e)^k
    int64_t exponentSum = 0;
    for (auto &[symbol, exponent] : monomial.getSymbols()) {
      exponentSum += exponent;
      if (symbol.isRandomVariable()) {
        // multiply variance for exponent times
        // also calculate j!
        for (auto i = 1; i <= exponent; ++i) {
          currentVariance *= symbol.getVariance() * (i);
        }
      }
    }
    // multiply N^{j+k-1}
    if (exponentSum > 0) {
      for (auto i = 0; i != exponentSum - 1; ++i) {
        currentVariance *= ringDim;
      }
    }
    variance += currentVariance;
  }
  // return log-scale value
  return variance.getValue();
}

}  // namespace noise
}  // namespace heir
}  // namespace mlir
