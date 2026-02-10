// RUN: heir-opt --secretize='function=main' --torch-linalg-to-ckks='ciphertext-degree=4096' %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x4x8x8xf32> {
    %cst = arith.constant dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_4_3_3_3_torch.float32> : tensor<4x3x3x3xf32>
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x3x8x8xf32> to tensor<1x3x10x10xf32>
    %0 = tensor.empty() : tensor<1x4x8x8xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<4xf32>) outs(%0 : tensor<1x4x8x8xf32>) dimensions = [0, 2, 3]
    // CHECK-NOT: linalg.conv_2d_nchw_fchw
    // CHECK-NOT: linalg.conv_2d
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %cst_1 : tensor<1x3x10x10xf32>, tensor<4x3x3x3xf32>) outs(%broadcasted : tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
    return %1 : tensor<1x4x8x8xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_4_torch.float32: "0x0400000009A02FBE0A24B13D3476023EDF51623B",
      torch_tensor_4_3_3_3_torch.float32: "0x04000000908C93BB74D132BD9878C83D44B8523D467EC3BD078C4B3D1B57ADBD1875283E892127BE481B083E64E99B3D4D04083E0D0A7DBD01D83C3EE8985EBD094E013EB16E343D7E31293DF2E396BD8E1AD6BA35ECB6BD9C860FBEB386EBBD0E1E9BBDAE0BEA3B07783FBEF289B7BD4BA8EBBB084DB3BC0133FBBCB996B6BD749F1B3D194337BB02F2C1BCB115833D3547613DF3B3B23DA13A0A3EE10EF5BAD9A9CD3DC85EFABC6AAC9ABD61ADFB3D5BB32ABEAF712D3ED5E844BEDD00B23C48DD0FBE8E4A1ABD275FAD3DEED970B8C6C4C03C740CE03DBA3EF33DB473F7BD22F910BED846273E82CA093E5D0536BD7CA5FE3CEC5D50BD64DD373EBA50853D440CE4BD067E23BE7A4062BDF1B82B3D486C22BEF9E61BBE6EFB0F3E98500EBEB9FC383E223FADBDCE8F44BE610D2E3EA57D1DBD81153DBED3A240BEAA8E943DBB40223EE0B12ABE89F470BD1A06BC3D1E8842BDCB4CB9BDA4593B384458133E202E08BE4DED923D68DE37BD0F2A0F3E38D1213D475904BE10E5FABCD20EF33D49EB2CBE300813BE47D5EDBD7040B1BD05B0353E0D86A6BC8984B43D2635223CAE4A113E5320153DE7F6E33CF82991BCD4FA0F3E"
    }
  }
#-}
