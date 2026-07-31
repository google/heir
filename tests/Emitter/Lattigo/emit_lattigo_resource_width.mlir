// externalize-constants writes DenseElementsAttr raw data, which stores f16 as
// 2 bytes per element, but convertType maps f16 to Go float32 and the generated
// loader reads 4 bytes per element. The emitter must refuse rather than emit a
// loader that cannot read the file the pass wrote.
// RUN: not heir-translate %s --emit-lattigo 2>&1 | FileCheck %s

// CHECK: resource element type 'f16' is stored as 2 bytes per element, but the generated loader reads float32
module attributes {scheme.bgv} {
  func.func @f16_resource() -> memref<4xf16> {
    %0 = preprocessing.load_resource "p/f16.bin" : memref<4xf16>
    return %0 : memref<4xf16>
  }
}
