#include "include/Dialect/Poly/IR/PolyAttributes.h"

#include "include/Dialect/Poly/IR/Polynomial.h"
#include "llvm/include/llvm/ADT/SmallSet.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"       // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace poly {

void PolynomialAttr::print(AsmPrinter &p) const {
  p << '<';
  p << getPolynomial();
  p << '>';
}

/// Try to parse a monomial. If successful, populate the fields of the outparam
/// `monomial` with the results, and the `variable` outparam with the parsed
/// variable name.
ParseResult parseMonomial(AsmParser &parser, Monomial &monomial,
                          llvm::StringRef *variable, bool *isConstantTerm) {
  APInt parsedCoeff(APINT_BIT_WIDTH, 1);
  auto result = parser.parseOptionalInteger(parsedCoeff);
  if (result.has_value()) {
    if (failed(*result)) {
      parser.emitError(parser.getCurrentLocation(),
                       "Invalid integer coefficient.");
      return failure();
    }
  }

  // Variable name
  result = parser.parseOptionalKeyword(variable);
  if (!result.has_value() || failed(*result)) {
    // we allow "failed" because it triggers when the next token is a +,
    // which is allowed when the input is the constant term.
    monomial.coefficient = parsedCoeff;
    monomial.exponent = APInt(APINT_BIT_WIDTH, 0);
    *isConstantTerm = true;
    return success();
  }

  // Parse exponentiation symbol as **
  // We can't use caret because it's reserved for basic block identifiers
  // If no star is present, it's treated as a polynomial with exponent 1
  if (failed(parser.parseOptionalStar())) {
    monomial.coefficient = parsedCoeff;
    monomial.exponent = APInt(APINT_BIT_WIDTH, 1);
    return success();
  }

  // If there's one * there must be two
  if (failed(parser.parseStar())) {
    parser.emitError(parser.getCurrentLocation(),
                     "Exponents must be specified as a double-asterisk `**`.");
    return failure();
  }

  // If there's a **, then the integer exponent is required.
  APInt parsedExponent(APINT_BIT_WIDTH, 0);
  if (failed(parser.parseInteger(parsedExponent))) {
    parser.emitError(parser.getCurrentLocation(),
                     "Found invalid integer exponent.");
    return failure();
  }

  monomial.coefficient = parsedCoeff;
  monomial.exponent = parsedExponent;
  return success();
}

mlir::Attribute mlir::heir::poly::PolynomialAttr::parse(AsmParser &parser,
                                                        Type type) {
  if (failed(parser.parseLess())) return {};

  std::vector<Monomial> monomials;
  llvm::SmallSet<std::string, 2> variables;
  llvm::DenseSet<APInt> exponents;

  while (true) {
    Monomial parsedMonomial;
    llvm::StringRef parsedVariableRef;
    bool isConstantTerm = false;
    if (failed(parseMonomial(parser, parsedMonomial, &parsedVariableRef,
                             &isConstantTerm))) {
      return {};
    }

    if (!isConstantTerm) {
      std::string parsedVariable = parsedVariableRef.str();
      variables.insert(parsedVariable);
    }
    monomials.push_back(parsedMonomial);

    if (exponents.count(parsedMonomial.exponent) > 0) {
      llvm::SmallString<512> coeff_string;
      parsedMonomial.exponent.toStringSigned(coeff_string);
      parser.emitError(parser.getCurrentLocation(),
                       "At most one monomial may have exponent " +
                           coeff_string + ", but found multiple.");
      return {};
    }
    exponents.insert(parsedMonomial.exponent);

    // Parse optional +. If a + is absent, require > and break, otherwise forbid
    // > and continue with the next monomial.
    // ParseOptional{Plus, Greater} does not return an OptionalParseResult, so
    // failed means that the token was not found.
    if (failed(parser.parseOptionalPlus())) {
      if (succeeded(parser.parseGreater())) {
        break;
      } else {
        parser.emitError(
            parser.getCurrentLocation(),
            "Expected + and more monomials, or > to end polynomial attribute.");
        return {};
      }
    } else if (succeeded(parser.parseOptionalGreater())) {
      parser.emitError(
          parser.getCurrentLocation(),
          "Expected another monomial after +, but found > ending attribute.");
      return {};
    }
  }

  if (variables.size() > 1) {
    std::string vars = llvm::join(variables.begin(), variables.end(), ", ");
    parser.emitError(
        parser.getCurrentLocation(),
        "Polynomials must have one indeterminate, but there were multiple: " +
            vars);
  }

  Polynomial poly =
      Polynomial::fromMonomials(std::move(monomials), parser.getContext());
  return PolynomialAttr::get(poly);
}

void RingAttr::print(AsmPrinter &p) const {
  p << "<cmod=";
  p << coefficientModulus();
  p << ", ideal=";
  p << PolynomialAttr::get(ideal());
  p << '>';
}

mlir::Attribute mlir::heir::poly::RingAttr::parse(AsmParser &parser,
                                                  Type type) {
  if (failed(parser.parseLess())) return {};

  if (failed(parser.parseKeyword("cmod"))) return {};

  if (failed(parser.parseEqual())) return {};

  APInt cmod(APINT_BIT_WIDTH, 0);
  auto result = parser.parseInteger(cmod);
  if (failed(result)) {
    parser.emitError(parser.getCurrentLocation(),
                     "Invalid coefficient modulus.");
    return {};
  }

  if (failed(parser.parseComma())) return {};

  if (failed(parser.parseKeyword("ideal"))) return {};

  if (failed(parser.parseEqual())) return {};

  PolynomialAttr polyAttr;
  if (failed(parser.parseAttribute<PolynomialAttr>(polyAttr))) return {};

  if (failed(parser.parseGreater())) return {};

  return RingAttr::get(cmod, polyAttr.getPolynomial());
}

}  // namespace poly
}  // namespace heir
}  // namespace mlir
