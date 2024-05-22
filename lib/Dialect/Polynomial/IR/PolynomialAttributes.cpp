#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"

#include <string>
#include <utility>
#include <vector>

#include "lib/Dialect/Polynomial/IR/NumberTheory.h"
#include "lib/Dialect/Polynomial/IR/Polynomial.h"
#include "llvm/include/llvm/ADT/SmallSet.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"       // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

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

mlir::Attribute mlir::heir::polynomial::PolynomialAttr::parse(AsmParser &parser,
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

  // insert necessary constant ops to give as input to extract_slice.
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
  coefficientModulus().print(p.getStream(), /*isSigned=*/false);
  p << ", ideal=";
  p << PolynomialAttr::get(ideal());
  if (primitive2NthRoot()) {
    p << ", root=";
    primitive2NthRoot()->print(p.getStream(), /*isSigned=*/false);
  }
  p << '>';
}

mlir::Attribute mlir::heir::polynomial::RingAttr::parse(AsmParser &parser,
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
  Polynomial poly = polyAttr.getPolynomial();

  unsigned rootBitWidth = (cmod - 1).getActiveBits();
  APInt root(rootBitWidth, 0);
  bool hasRoot = succeeded(parser.parseOptionalComma());
  if (hasRoot) {
    if (failed(parser.parseKeyword("root"))) return {};

    if (failed(parser.parseEqual())) return {};

    auto result = parser.parseInteger(root);
    if (failed(result) || cmod.ule(root.zext(cmod.getBitWidth())) ||
        !isPrimitiveNthRootOfUnity(root, 2 * poly.getDegree(), cmod)) {
      parser.emitError(parser.getCurrentLocation(),
                       "Invalid 2n-th primitive root of unity.");
      return {};
    }
  }

  if (failed(parser.parseGreater())) return {};

  return hasRoot ? RingAttr::get(cmod, poly, root) : RingAttr::get(cmod, poly);
}

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
