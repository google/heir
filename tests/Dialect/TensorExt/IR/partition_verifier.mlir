// RUN: heir-opt --verify-diagnostics --split-input-file %s

func.func @fn(%0: tensor<17xi32>) -> () {
  // expected-error@+1 {{output must have 1 or 2 types}}
  %1:3 = tensor_ext.partition %0 {partitionSize = 5 : index} : tensor<17xi32> -> (tensor<1xi32>, tensor<2xi32>, tensor<3xi32>)
  return
}

// -----

func.func @fn(%0: tensor<17x17xi32>) -> () {
  // expected-error@+1 {{input tensor must have rank 1}}
  %1:2 = tensor_ext.partition %0 {partitionSize = 5 : index} : tensor<17x17xi32> -> (tensor<3x5xi32>, tensor<6xi32>)
  return
}

// -----

func.func @fn(%0: tensor<15xi32>) -> () {
  // expected-error@+1 {{element types must match}}
  %1 = tensor_ext.partition %0 {partitionSize = 5 : index} : tensor<15xi32> -> (tensor<3x5xi64>)
  return
}

// -----

func.func @fn(%0: tensor<17xi32>) -> (tensor<3x5xi32>, tensor<6xi32>) {
  // expected-error@+1 {{trailing component has incorrect shape}}
  %1:2 = tensor_ext.partition %0 {partitionSize = 5 : index} : tensor<17xi32> -> (tensor<3x5xi32>, tensor<6xi32>)
  return %1#0, %1#1 : tensor<3x5xi32>, tensor<6xi32>
}

// -----

func.func @fn(%0: tensor<17xi32>) {
  // expected-error@+1 {{partition result has incorrect shape}}
  %1:2 = tensor_ext.partition %0 {partitionSize = 5 : index} : tensor<17xi32> -> (tensor<3x6xi32>, tensor<2xi32>)
  return
}

// -----

func.func @fn(%0: tensor<17xi32>) -> tensor<3x5xi32> {
  // expected-error@+1 {{missing required trailing component}}
  %1 = tensor_ext.partition %0 {partitionSize = 5 : index} : tensor<17xi32> -> (tensor<3x5xi32>)
  return %1 : tensor<3x5xi32>
}

// -----

func.func @fn(%0: tensor<15xi32>) -> (tensor<3x5xi32>, tensor<2xi32>) {
  // expected-error@+1 {{output contains unneeded trailing component}}
  %1:2 = tensor_ext.partition %0 {partitionSize = 5 : index} : tensor<15xi32> -> (tensor<3x5xi32>, tensor<2xi32>)
  return %1#0, %1#1 : tensor<3x5xi32>, tensor<2xi32>
}
