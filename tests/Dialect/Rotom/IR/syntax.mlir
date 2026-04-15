// RUN: heir-opt %s

#d0 = #rotom.dim<dim = 0, size = 4, stride = 1>
#d1 = #rotom.dim<dim = 1, size = 4, stride = 1>
#layout = #rotom.layout<dims = [#d0, #d1], n = 16>
module {
  func.func @f(%arg0: tensor<4x4xf32> {rotom.layout = #layout}) {
    return
  }
}
