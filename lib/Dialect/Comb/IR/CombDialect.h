//===- CombDialect.h - Comb dialect declaration -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Combinational MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HEIR_LIB_DIALECT_COMB_COMBDIALECT_H
#define HEIR_LIB_DIALECT_COMB_COMBDIALECT_H

#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"            // from @llvm-project

// Pull in the Dialect definition.
#include "lib/Dialect/Comb/IR/CombDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "lib/Dialect/Comb/IR/CombEnums.h.inc"

#endif  // HEIR_LIB_DIALECT_COMB_COMBDIALECT_H
