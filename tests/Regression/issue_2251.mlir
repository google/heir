module {
  func.func @main(%input: tensor<25xi32> ,
                %mat: tensor<15x25xi32>) -> tensor<15xi32> {
    %c_0_idx = arith.constant 0 : index
    %c_one = arith.constant 1 : index
    %c_0_i32 = arith.constant 0 : i32
    %c_15 = arith.constant 15 : index
    %c_25 = arith.constant 25 : index

    %empty = tensor.empty() : tensor<15xi32>

    %layer1_out = scf.for %i = %c_0_idx to %c_15 step %c_one iter_args(%out = %empty) -> tensor<15xi32> {
      %acc_final = scf.for %j = %c_0_idx to %c_25 step %c_one iter_args(%acc = %c_0_i32) -> (i32) {
        %val = tensor.extract %mat[%i, %j] : tensor<15x25xi32>
        %input_val = tensor.extract %input[%j] : tensor<25xi32>
        %prod = arith.muli %val, %input_val : i32
        %acc_next = arith.addi %acc, %prod : i32
        scf.yield %acc_next : i32
      }

      %out_next = tensor.insert %acc_final into %out[%i] : tensor<15xi32>
      scf.yield %out_next : tensor<15xi32>
    }

    func.return %layer1_out : tensor<15xi32>
  }
}
