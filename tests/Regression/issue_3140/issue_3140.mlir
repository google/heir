module {
  func.func @issue_3140(%arg0: i16 {secret.secret}, %arg1: i16 {secret.secret}) -> (i16 {secret.secret}) {
    %0 = arith.muli %arg0, %arg1 {secret.secret} : i16
    %c1_i8 = arith.constant 1 : i8
    %1 = arith.extsi %c1_i8 : i8 to i16
    %2 = arith.addi %0, %1 {secret.secret} : i16
    return {secret.secret} %2 : i16
  }
}
