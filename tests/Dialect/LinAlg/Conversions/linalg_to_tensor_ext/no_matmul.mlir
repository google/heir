// RUN: heir-opt --wrap-generic --linalg-to-tensor-ext %s | FileCheck %s

// CHECK-LABEL: func @add
func.func @add(%arg0 : i16 {secret.secret}, %arg1 : i16 {secret.secret}) -> i16 {
    return %arg0 : i16
}
