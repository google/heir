#include <alloca.h>

#include <cstdint>
#include <cstring>

// Copied from CRunnerUtils from MLIR

/// StridedMemRef descriptor type with static rank.
template <typename T, int N>
struct StridedMemRefType {
  T* basePtr;
  T* data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

//===----------------------------------------------------------------------===//
// Codegen-compatible structure for UnrankedMemRef type.
//===----------------------------------------------------------------------===//
// Unranked MemRef
template <typename T>
struct UnrankedMemRefType {
  int64_t rank;
  void* descriptor;
};

// A reference to one of the StridedMemRef types.
template <typename T>
class DynamicMemRefType {
 public:
  int64_t rank;
  T* basePtr;
  T* data;
  int64_t offset;
  const int64_t* sizes;
  const int64_t* strides;

  explicit DynamicMemRefType(const ::UnrankedMemRefType<T>& memRef)
      : rank(memRef.rank) {
    auto* desc = static_cast<StridedMemRefType<T, 1>*>(memRef.descriptor);
    basePtr = desc->basePtr;
    data = desc->data;
    offset = desc->offset;
    sizes = rank == 0 ? nullptr : desc->sizes;
    strides = sizes + rank;
  }
};

extern "C" void memrefCopy(int64_t elemSize, UnrankedMemRefType<char>* srcArg,
                           UnrankedMemRefType<char>* dstArg) {
  DynamicMemRefType<char> src(*srcArg);
  DynamicMemRefType<char> dst(*dstArg);

  int64_t rank = src.rank;
  // MLIR_MSAN_MEMORY_IS_INITIALIZED(src.sizes, rank * sizeof(int64_t));

  // Handle empty shapes -> nothing to copy.
  for (int rankp = 0; rankp < rank; ++rankp)
    if (src.sizes[rankp] == 0) return;

  char* srcPtr = src.data + src.offset * elemSize;
  char* dstPtr = dst.data + dst.offset * elemSize;

  if (rank == 0) {
    memcpy(dstPtr, srcPtr, elemSize);
    return;
  }

  int64_t* indices = static_cast<int64_t*>(alloca(sizeof(int64_t) * rank));
  int64_t* srcStrides = static_cast<int64_t*>(alloca(sizeof(int64_t) * rank));
  int64_t* dstStrides = static_cast<int64_t*>(alloca(sizeof(int64_t) * rank));

  // Initialize index and scale strides.
  for (int rankp = 0; rankp < rank; ++rankp) {
    indices[rankp] = 0;
    srcStrides[rankp] = src.strides[rankp] * elemSize;
    dstStrides[rankp] = dst.strides[rankp] * elemSize;
  }

  int64_t readIndex = 0, writeIndex = 0;
  for (;;) {
    // Copy over the element, byte by byte.
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, elemSize);
    // Advance index and read position.
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      // Advance at current axis.
      auto newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];
      // If this is a valid index, we have our next index, so continue copying.
      if (src.sizes[axis] != newIndex) break;
      // We reached the end of this axis. If this is axis 0, we are done.
      if (axis == 0) return;
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      readIndex -= src.sizes[axis] * srcStrides[axis];
      writeIndex -= dst.sizes[axis] * dstStrides[axis];
    }
  }
}
