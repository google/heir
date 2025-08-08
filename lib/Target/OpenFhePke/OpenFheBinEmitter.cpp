#include "lib/Target/OpenFhePke/OpenFheBinEmitter.h"

#include <numeric>
#include <set>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Comb/IR/CombDialect.h"
#include "lib/Dialect/Comb/IR/CombOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir::heir::openfhe {

// clang-format off
constexpr std::string_view prelude = R"cpp(
#include "src/binfhe/include/binfhecontext.h"  // from @openfhe
#include "src/binfhe/include/lwe-privatekey.h"  // from @openfhe
#include "src/binfhe/include/lwe-publickey.h"  // from @openfhe
#include "src/binfhe/include/lwe-ciphertext.h"  // from @openfhe
#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/core/include/math/math-hal.h"  // from @openfhe

#include <algorithm>
#include <utility>
#include <vector>

using namespace lbcrypto;

using BinFHEContextT = std::shared_ptr<BinFHEContext>;
using LWESchemeT = std::shared_ptr<LWEEncryptionScheme>;
using CiphertextT = LWECiphertext;

constexpr int ptxt_mod = 8;

std::vector<LWECiphertext> encrypt(BinFHEContextT cc, LWEPrivateKey sk,
                                    int value, int width) {
  std::vector<lbcrypto::LWECiphertext> encrypted_bits;
  for (int i = 0; i < width; i++) {
    int bit = (value & (1 << i)) >> i;
    encrypted_bits.push_back(
        cc->Encrypt(sk, bit, BINFHE_OUTPUT::SMALL_DIM, ptxt_mod));
  }
  return encrypted_bits;
}

int decrypt(BinFHEContextT cc, LWEPrivateKey sk,
            std::vector<LWECiphertext> encrypted) {
  int result = 0;

  std::reverse(encrypted.begin(), encrypted.end());

  for (LWECiphertext encrypted_bit : encrypted) {
    LWEPlaintext bit;
    cc->Decrypt(sk, encrypted_bit, &bit, ptxt_mod);
    result *= 2;
    result += bit;
  }
  return result;
}

LWECiphertext copy(LWECiphertext ctxt) {
  LWECiphertext copied =
      std::make_shared<LWECiphertextImpl>(ctxt->GetA(), ctxt->GetB());
  // Preserve plaintext modulus carried by the ciphertext; default is 4.
  copied->SetptModulus(ctxt->GetptModulus());
  return copied;
}

// I'm done apologizing...
struct view_t {
    int offset;
    int stride;
    int size;

    view_t(int offset, int stride, int size) : offset(offset), stride(stride), size(size) {}
    void apply(view_t other) {
        offset += other.offset * stride;
        stride *= other.stride;
        size = other.size;
    }
};

template<class T>
constexpr int dim = 0;

template<class T>
constexpr int dim<std::vector<T>> = 1 + dim<T>;

template<class T>
constexpr int dim<std::vector<T>&> = 1 + dim<T>;

template<class T, int rank>
struct of_rank {
    using type = std::vector<typename of_rank<T, rank - 1>::type>;
};

template<class T>
struct of_rank<T, 0> {
    using type = T;
};

template <class T>
struct underlying {
    using type = T;
};

template <class T>
struct underlying<std::vector<T>> {
    using type = typename underlying<T>::type;
};

template <class T>
class vector_view;

template <class T>
class vector_view<std::vector<T>> {
    std::vector<T>& data;
    std::vector<T> owned_data;
    std::vector<view_t> views;
public:
    using underlying_t = typename underlying<T>::type;
    vector_view(std::vector<T>& data, std::vector<view_t> views) : data(data), views(views) {}
    vector_view(std::vector<T>& data) : data(data) {
        for (int i = 0; i < dim<std::vector<T>>; i++) {
            views.emplace_back(0, 1, -1);
        }
    }
    vector_view(const std::vector<T>& elems) : data(owned_data), owned_data(elems) {
        for (int i = 0; i < dim<decltype(data)>; i++) {
            views.emplace_back(0, 1, -1);
        }
    }

    vector_view(const vector_view<T>& other) : data(other.data), views(other.views) {}

    constexpr int rank() {
        return dim<decltype(data)>;
    }

    size_t size() const {
        if (views[0].size == -1) return data.size();
        return views[0].size;
    }

    auto operator[](size_t index) -> decltype(auto) {
        auto& vec = data[views[0].offset + index * views[0].stride];
        if constexpr (dim<decltype(data)> == 1) {
            return vec;
        } else {
            vector_view<T> new_view(vec, std::vector<view_t>(views.begin() + 1, views.end()));
            return new_view;
        }
    }

    auto operator[](size_t index) const -> decltype(auto) {
        auto& vec = data[views[0].offset + index * views[0].stride];
        if constexpr (dim<decltype(data)> == 1) {
            return vec;
        } else {
            vector_view<T> new_view(vec, std::vector<view_t>(views.begin() + 1, views.end()));
            return new_view;
        }
    }

    auto subview(std::vector<view_t> slices) {
        auto copied = *this;
        for (int i = 0; i < slices.size(); i++) {
            copied.views[i].apply(slices[i]);
        }
        return copied;
    }

    std::vector<T> copy() const {
        std::vector<T> copied;
        for (size_t i = 0; i < size(); i++) {
            if constexpr (dim<decltype(data)> == 1) {
                copied.push_back(this->operator[](i));
            } else {
                copied.push_back(this->operator[](i).copy());
            }
        }
        return copied;
    }

    void flatten_into(const std::vector<underlying_t>& dest, int *index = nullptr) const {
        int local_index = 0;
        if (index == nullptr) {
            index = &local_index;
        }
        for (size_t i = 0; i < size(); i++) {
            if constexpr (dim<decltype(data)> == 1) {
                dest[(*index)++] = this->operator[](i);
            } else {
                this->operator[](i).flatten_into(dest, index);
            }
        }
    }

    void flatten(std::vector<underlying_t>& dest) const {
        for (size_t i = 0; i < size(); i++) {
            if constexpr (dim<decltype(data)> == 1) {
                dest.push_back(this->operator[](i));
            } else {
                this->operator[](i).flatten(dest);
            }
        }
    }

    void unflatten_from(std::vector<underlying_t>& dest, int *index = nullptr) {
        int local_index = 0;
        if (index == nullptr) {
            index = &local_index;
        }

        for (size_t i = 0; i < size(); i++) {
            if constexpr (dim<decltype(data)> == 1) {
                this->operator[](i) = dest[(*index)++];
            } else {
                this->operator[](i).unflatten_from(dest, index);
            }
        }
    }

};

template <class T>
class vector_view<std::vector<T>&> : public vector_view<std::vector<T>> {
public:
    vector_view(std::vector<T>& data) : vector_view<std::vector<T>>(data) {}
};

LWECiphertext trivialEncrypt(BinFHEContextT cc, LWEPlaintext ptxt) {
  auto params = cc->GetParams()->GetLWEParams();

  NativeVector a(params->Getn(), params->Getq(), 0);
  NativeInteger b(ptxt * (params->Getq() / ptxt_mod));
  return std::make_shared<LWECiphertextImpl>(a, b);
}

template <class T, int dim>
std::vector<T> unflatten(std::vector<T> source, int start = 0) {
    std::vector<T> result;
    for (int i = start; i < start + dim; i++)
        result.push_back(source[i]);
    return result;
}

template <class T, int firstDim, int... restDims>
auto unflatten(std::vector<T> source, int start = 0) -> std::vector<decltype(unflatten<T, restDims...>(std::declval<std::vector<T>>(), std::declval<int>()))> {
    using return_type = typename of_rank<T, sizeof...(restDims) + 1>::type;
    return_type unflattened;

    int stride = (1 * ... * restDims);
    for (int i = 0; i < firstDim; i++) {
        unflattened.push_back(unflatten<T, restDims...>(source, start));
        start += stride;
    }
    return unflattened;
}

template <class underlying, int rank>
class iterator {
    vector_view<typename of_rank<underlying, rank>::type> *view;
    int index;
    bool done = false;
    iterator<underlying, rank - 1> inner_iter;

public:
    iterator(const vector_view<typename of_rank<underlying, rank>::type>& view, int index) :
        view(new vector_view(view)), index(index), inner_iter(view[0], 0) {}
    underlying& operator*() {
        return *inner_iter;
    }

    iterator(const iterator<underlying, rank>& other) :
        view(new vector_view(other.view)), inner_iter(other.inner_iter), index(index) {}

    iterator<underlying, rank>& operator=(const iterator<underlying, rank>& other) {
        auto copy = other;
        std::swap(view, copy.view);
        std::swap(index, copy.index);
        std::swap(inner_iter, copy.inner_iter);
        return *this;
    }

    iterator<underlying, rank>& operator++() {
        // std::cout << "Incrementing iterator of rank " << rank << "\n";
        ++inner_iter;
        if (inner_iter.at_end()) {
            if (++index == view->size()) {
                done = true;
            } else {
                inner_iter = iterator<underlying, rank - 1>(view->operator[](index), 0);
            }

        }
        return *this;
    }

    bool at_end() {
        return done;
    }
};

template <class underlying>
class iterator<underlying, 1> {
    vector_view<std::vector<underlying>> *view;
    int index;

public:
    iterator(const vector_view<std::vector<underlying>>& view, int index) : view(new vector_view(view)), index(index) {}
    iterator(const iterator<underlying, 1>& other) :
        view(new vector_view(*other.view)), index(other.index) {}

    iterator<underlying, 1>& operator=(const iterator<underlying, 1>& other) {
        auto copy = other;
        std::swap(view, copy.view);
        std::swap(index, copy.index);
        return *this;
    }

    underlying& operator*() {
        return view->operator[](index);
    }
    iterator<underlying, 1>& operator++() {
        // std::cout << "Advancing through a vector\n";
        index++;
        return *this;
    }
    bool at_end() {
        return index == view->size();
    }
};

template<class T>
auto begin(const vector_view<T>& view) {
    using U = typename underlying<T>::type;
    constexpr int rank = dim<T>;
    return iterator<U, rank>(view, 0);
}

template<class T>
auto end(const vector_view<T>& view) {
    using U = typename underlying<T>::type;
    constexpr int rank = dim<T>;
    return iterator<U, rank>(view, view.size());
}

template <class S, class T>
void copy(const vector_view<S>& dest, const vector_view<T>& source) {
    auto i = begin(dest);
    auto j = begin(source);

    for (; !(i.at_end() || j.at_end()); ++i, ++j) {
        *i = *j;
    }
}

)cpp";
// clang-format on

// static llvm::SmallVector<std::pair<int, int>> getIntervals(
//     const llvm::ArrayRef<int> &values) {
//   llvm::SmallVector<std::pair<int, int>> intervals;
//   std::pair<int, int> current{values[0], values[0]};
//   for (int value : values) {
//     if (value == current.second + 1) {
//       current.second = value;
//     } else if (value != current.second) {
//       intervals.push_back(current);
//       current = {value, value};
//     }
//   }
//   if (intervals.end()->first != current.first) intervals.push_back(current);
//   return intervals;
// }

// Registration is done in OpenFheTranslateRegistration.cpp

LogicalResult translateToOpenFheBin(mlir::Operation *op,
                                    llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  OpenFheBinEmitter emitter(os, &variableNames);

  return emitter.translate(*op);
}

LogicalResult OpenFheBinEmitter::translate(Operation &operation) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(operation)
          .Case<mlir::ModuleOp>(
              [&](auto module) { return printOperation(module); })
          // Func ops
          .Case<func::FuncOp, func::CallOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp>([&](auto op) { return printOperation(op); })
          // Tensor ops
          .Case<tensor::ExtractOp, tensor::FromElementsOp>(
              [&](auto op) { return printOperation(op); })
          .Case<memref::LoadOp>([&](auto load) { return printOperation(load); })
          .Case<memref::StoreOp>(
              [&](auto store) { return printOperation(store); })
          .Case<memref::SubViewOp>(
              [&](auto subview) { return printOperation(subview); })
          .Case<memref::ReinterpretCastOp>(
              [&](auto castOp) { return printOperation(castOp); })
          .Case<memref::CopyOp>([&](auto copy) { return printOperation(copy); })
          .Case<memref::CollapseShapeOp>(
              [&](auto collapse) { return printOperation(collapse); })
          .Case<memref::AllocOp>(
              [&](auto alloc) { return printOperation(alloc); })
          .Case<openfhe::GetLWESchemeOp>(
              [&](auto getScheme) { return printOperation(getScheme); })
          .Case<openfhe::LWEMulConstOp>(
              [&](auto mul) { return printOperation(mul); })
          .Case<openfhe::LWEAddOp>(
              [&](auto add) { return printOperation(add); })
          .Case<openfhe::MakeLutOp>(
              [&](auto makeLut) { return printOperation(makeLut); })
          .Case<openfhe::EvalFuncOp>(
              [&](auto evalFunc) { return printOperation(evalFunc); })
          .Case<lwe::EncodeOp, lwe::TrivialEncryptOp>(
              [&](auto op) { return printOperation(op); })
          .Case<scf::IfOp>([&](auto ifOp) { return printOperation(ifOp); })
          .Case<affine::AffineForOp>(
              [&](auto forOp) { return printOperation(forOp); })
          .Case<comb::InvOp>([&](auto invOp) { return printOperation(invOp); })
          .Default([&](auto &op) { return translateFallback(operation); });
  return status;
}

LogicalResult OpenFheBinEmitter::printOperation(mlir::ModuleOp module) {
  os << prelude << "\n";
  for (Operation &op : module) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(memref::LoadOp load) {
  if (failed(emitTypedAssignPrefix(load.getResult(), load->getLoc()))) {
    return failure();
  }
  os << variableNames->getNameForValue(load.getMemRef()) << "["
     << variableNames->getNameForValue(load.getIndices()[0]) << "];\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(memref::StoreOp store) {
  os << variableNames->getNameForValue(store.getMemRef()) << "[";
  os << variableNames->getNameForValue(store.getIndices()[0]);
  os << "] = " << variableNames->getNameForValue(store.getValueToStore())
     << ";\n";
  return success();
}

mlir::FailureOr<std::string> OpenFheBinEmitter::getAllocConstructor(
    MemRefType type, Location loc) {
  std::string output;
  llvm::raw_string_ostream ss(output);

  auto typeResult = convertType(type, loc);
  if (failed(typeResult)) {
    return failure();
  }

  ss << typeResult.value() << "(" << type.getShape()[0];
  if (type.getRank() > 1) {
    auto sliced =
        MemRefType::get({type.getShape().begin() + 1, type.getShape().end()},
                        type.getElementType());
    auto rest = getAllocConstructor(sliced, loc);
    if (failed(rest)) {
      return failure();
    }
    ss << ", " << rest.value();
  }
  ss << ")";

  return output;
}

LogicalResult OpenFheBinEmitter::printOperation(memref::AllocOp alloc) {
  auto memrefType = alloc.getResult().getType();
  if (failed(emitTypedAssignPrefix(alloc.getResult(), alloc->getLoc()))) {
    return failure();
  }

  auto allocResult = getAllocConstructor(memrefType, alloc->getLoc());
  if (failed(allocResult)) {
    return failure();
  }

  os << allocResult << ";\n";
  return success();

  // const auto *shapeBegin = memrefType.getShape().begin();
  // for (int dim = 0; dim < memrefType.getRank(); dim++) {
  //   auto rankType = MemRefType::get({shapeBegin,
  //   memrefType.getShape().end()}, memrefType.getElementType()); auto
  //   typeResult = convertType(rankType); if (failed(typeResult)) return
  //   failure(); os << typeResult.value() << "(" << *shapeBegin;
  // }

  // auto typeResult = convertType(*alloc->getResultTypes().begin());
  // if (failed(typeResult)) return failure();
  // os << typeResult.value() << " ";
  // os << variableNames->getNameForValue(alloc.getResult()) << "(";
  // os << alloc.getResult().getType().getShape()[0] << ");\n";
  // return success();
}

LogicalResult OpenFheBinEmitter::printOperation(lwe::EncodeOp encode) {
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(
    lwe::TrivialEncryptOp trivialEncrypt) {
  Value cryptoContext = trivialEncrypt->getParentOfType<func::FuncOp>()
                            .getBody()
                            .getBlocks()
                            .front()
                            .getArguments()
                            .front();

  if (auto encoder =
          dyn_cast<lwe::EncodeOp>(trivialEncrypt.getInput().getDefiningOp())) {
    auto ptxtName = variableNames->getNameForValue(encoder.getInput());
    emitAutoAssignPrefix(trivialEncrypt.getResult());
    os << "trivialEncrypt(" << variableNames->getNameForValue(cryptoContext)
       << ", " << ptxtName << ");\n";
    return success();
  }
  return failure();
}

LogicalResult OpenFheBinEmitter::printOperation(comb::InvOp op) {
  emitAutoAssignPrefix(op.getResult());
  os << "!" << variableNames->getNameForValue(op.getInput());
  return success();
}

SmallVector<std::string> OpenFheBinEmitter::getStaticDynamicArgs(
    SmallVector<mlir::Value> dynamicArgs, ArrayRef<int64_t> staticArgs) {
  SmallVector<std::string> args;
  int dynamicIndex = 0;
  for (int64_t staticArg : staticArgs) {
    if (staticArg == ShapedType::kDynamic) {
      args.push_back(
          variableNames->getNameForValue(dynamicArgs[dynamicIndex++]));
    } else {
      args.push_back(std::to_string(staticArg));
    }
  }
  return args;
}

template <class T, typename>
std::string OpenFheBinEmitter::getSubviewArgs(T op) {
  SmallVector<std::string> offsets =
      getStaticDynamicArgs(op.getOffsets(), op.getStaticOffsets());
  SmallVector<std::string> strides =
      getStaticDynamicArgs(op.getStrides(), op.getStaticStrides());
  SmallVector<std::string> sizes =
      getStaticDynamicArgs(op.getSizes(), op.getStaticSizes());

  SmallVector<std::string> viewStrings;
  for (size_t i = 0; i < offsets.size(); i++) {
    SmallString<8> viewString;
    llvm::raw_svector_ostream ss(viewString);
    ss << "view_t(" << offsets[i] << ", " << strides[i] << ", " << sizes[i]
       << ")";
    viewStrings.push_back(viewString.str().str());
  }
  std::string args = std::accumulate(
      std::next(viewStrings.begin()), viewStrings.end(), viewStrings[0],
      [&](const std::string &a, const std::string &b) { return a + ", " + b; });

  return args;
}

LogicalResult OpenFheBinEmitter::printOperation(memref::SubViewOp subview) {
  std::string args = getSubviewArgs(subview);
  emitAutoAssignPrefix(subview.getResult());
  std::string sourceName = variableNames->getNameForValue(subview.getSource());
  os << "vector_view<decltype(" << sourceName << ")>(" << sourceName
     << ").subview({" << args << "});\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(memref::CopyOp copy) {
  std::string source, dest;
  if (copy.getTarget().getDefiningOp<memref::SubViewOp>()) {
    dest = variableNames->getNameForValue(copy.getTarget());
  } else {
    dest = llvm::formatv("vector_view<{}>({})",
                         convertType(copy.getTarget().getType(), copy.getLoc()),
                         variableNames->getNameForValue(copy.getTarget()));
  }

  if (copy.getSource().getDefiningOp<memref::SubViewOp>()) {
    source = variableNames->getNameForValue(copy.getSource());
  } else {
    source =
        llvm::formatv("vector_view<{}>({})",
                      convertType(copy.getSource().getType(), copy->getLoc()),
                      variableNames->getNameForValue(copy.getSource()));
  }

  os << llvm::formatv("copy({}, {});\n", dest, source);
  // variableNames->getNameForValue(copy.getTarget())

  // if (copy.getSource().getType().getRank() >=
  //     copy.getTarget().getType().getRank()) {
  //   os << variableNames->getNameForValue(copy.getSource()) <<
  //   ".flatten_into("; os << variableNames->getNameForValue(copy.getTarget())
  //   << ");\n";
  // } else {
  //   llvm::dbgs() << "Unflattening from " << copy.getSource().getType() << "
  //   to " << copy.getTarget().getType() << "\n"; os <<
  //   variableNames->getNameForValue(copy.getTarget())
  //      << ".unflatten_from(";
  //   os << variableNames->getNameForValue(copy.getSource()) << ");\n";
  // }

  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(
    memref::ReinterpretCastOp castOp) {
  std::string sourceName = variableNames->getNameForValue(castOp.getSource());
  if (castOp.getSource().getType().getRank() >
      mlir::cast<BaseMemRefType>(castOp->getResult(0).getType()).getRank()) {
    std::string args = getSubviewArgs(castOp);
    os << convertType(castOp->getResult(0).getType(), castOp->getLoc()) << " "
       << variableNames->getNameForValue(castOp->getResult(0)) << ";\n";

    os << llvm::formatv(
        "vector_view<decltype({})>({})/*.subview({{{}})*/.flatten({});\n",
        sourceName, sourceName, args,
        variableNames->getNameForValue(castOp->getResult(0)));
  } else {
    emitAutoAssignPrefix(castOp->getResult(0));
    auto destinationShape =
        mlir::cast<BaseMemRefType>(castOp->getResult(0).getType()).getShape();
    os << "unflatten<"
       << convertType(castOp.getSource().getType().getElementType(),
                      castOp.getLoc())
       << ", "
       << std::accumulate(std::next(destinationShape.begin()),
                          destinationShape.end(),
                          std::to_string(destinationShape[0]),
                          [](auto dim1, auto dim2) {
                            return dim1 + ", " + std::to_string(dim2);
                          })
       << ">(" << sourceName << ");\n";
  }
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(
    memref::CollapseShapeOp collapseOp) {
  os << convertType(collapseOp->getResult(0).getType(), collapseOp->getLoc())
     << " " << variableNames->getNameForValue(collapseOp->getResult(0))
     << ";\n";
  std::string sourceName =
      variableNames->getNameForValue(collapseOp.getViewSource());
  os << llvm::formatv("vector_view<decltype({})>({}).flatten({});\n",
                      sourceName, sourceName,
                      variableNames->getNameForValue(collapseOp->getResult(0)));
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(
    openfhe::GetLWESchemeOp getScheme) {
  auto cryptoContext = getScheme.getCryptoContext();
  emitAutoAssignPrefix(getScheme.getResult());
  os << variableNames->getNameForValue(cryptoContext) << "->GetLWEScheme();\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::LWEMulConstOp mul) {
  return printInPlaceEvalMethod(mul.getResult(), mul.getCryptoContext(),
                                {mul.getCiphertext(), mul.getConstant()},
                                "EvalMultConstEq", mul->getLoc());
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::LWEAddOp add) {
  return printInPlaceEvalMethod(add.getResult(), add.getCryptoContext(),
                                {add.getOperand(1), add.getOperand(2)},
                                "EvalAddEq", add->getLoc());
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::MakeLutOp makeLut) {
  emitAutoAssignPrefix(makeLut.getOutput());

  // Create a set of values for quick lookup
  std::set<int32_t> valueSet;
  for (auto val : makeLut.getValues()) {
    valueSet.insert(val);
  }

  auto ccName = variableNames->getNameForValue(makeLut.getCryptoContext());
  os << ccName
     << "->GenerateLUTviaFunction([](NativeInteger m, NativeInteger p) -> "
        "NativeInteger {\n";

  // Generate the LUT based on which indices should output 1
  for (int i = 0; i < 8; i++) {
    if (valueSet.count(i)) {
      os << llvm::formatv("  if (m == {0}) return 1;\n", i);
    }
  }
  os << "  return 0;\n";
  os << "}, ptxt_mod);\n";

  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(openfhe::EvalFuncOp evalFunc) {
  return printEvalMethod(evalFunc.getResult(), evalFunc.getCryptoContext(),
                         {evalFunc.getInput(), evalFunc.getLut()}, "EvalFunc");
}

LogicalResult OpenFheBinEmitter::printOperation(func::FuncOp funcOp) {
  if (funcOp.getNumResults() > 1) {
    return emitError(
        funcOp.getLoc(),
        "Only functions with zero or one return types are supported");
  }

  // Generate function signature
  if (funcOp.getNumResults() == 1) {
    Type result = funcOp.getResultTypes()[0];
    if (failed(emitType(result, funcOp.getLoc()))) {
      return failure();
    }
  } else {
    os << "void";
  }

  os << " " << funcOp.getName() << "(";

  // Emit function arguments
  Block &entryBlock = funcOp.getBody().front();
  os << commaSeparatedValues(entryBlock.getArguments(), [&](Value arg) {
    auto typeStr = convertType(arg.getType(), funcOp.getLoc());
    if (failed(typeStr)) {
      return std::string("/* error */");
    }
    return typeStr.value() + " " + variableNames->getNameForValue(arg);
  });

  os << ") {\n";

  // Emit function body
  for (Operation &op : entryBlock.getOperations()) {
    os << "  ";
    if (failed(translate(op))) {
      return failure();
    }
  }

  os << "}\n\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(func::CallOp callOp) {
  if (callOp.getNumResults() > 1) {
    return emitError(
        callOp.getLoc(),
        "Only functions with zero or one return types are supported");
  }

  if (callOp.getNumResults() == 1) {
    emitAutoAssignPrefix(callOp.getResult(0));
  }

  os << callOp.getCallee() << "(";
  os << commaSeparatedValues(callOp.getOperands(), [&](Value operand) {
    return variableNames->getNameForValue(operand);
  });
  os << ");\n";

  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(func::ReturnOp returnOp) {
  if (returnOp.getNumOperands() == 0) {
    os << "return;\n";
  } else {
    os << "return " << variableNames->getNameForValue(returnOp.getOperand(0))
       << ";\n";
  }
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(arith::ConstantOp constantOp) {
  emitAutoAssignPrefix(constantOp.getResult());
  auto valueAttr = constantOp.getValue();

  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    os << intAttr.getInt();
  } else if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
    SmallString<128> strValue;
    auto apValue = APFloat(floatAttr.getValueAsDouble());
    apValue.toString(strValue, /*FormatPrecision=*/0, /*FormatMaxPadding=*/15,
                     /*TruncateZero=*/true);
    os << strValue;
  } else if (auto indexAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    os << indexAttr.getInt();
  } else {
    return emitError(constantOp.getLoc(), "unsupported constant type");
  }

  os << ";\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(tensor::ExtractOp extractOp) {
  emitAutoAssignPrefix(extractOp.getResult());
  os << variableNames->getNameForValue(extractOp.getTensor()) << "[";
  os << commaSeparatedValues(extractOp.getIndices(), [&](Value index) {
    return variableNames->getNameForValue(index);
  });
  os << "];\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(
    tensor::FromElementsOp fromElementsOp) {
  if (failed(emitTypedAssignPrefix(fromElementsOp.getResult(),
                                   fromElementsOp->getLoc()))) {
    return failure();
  }
  os << "{";
  os << commaSeparatedValues(fromElementsOp.getElements(), [&](Value element) {
    return variableNames->getNameForValue(element);
  });
  os << "};\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(scf::IfOp ifOp) {
  if (failed(emitType(ifOp->getResultTypes().front(), ifOp->getLoc()))) {
    return failure();
  }

  auto &thenBlock = *ifOp.thenBlock();
  auto &elseBlock = *ifOp.elseBlock();

  // Assumes the type is default-constructible (I don't have a good way around
  // this)
  auto resultName = variableNames->getNameForValue(ifOp->getResult(0));
  os << " " << resultName << ";\n";

  os << "if (" << variableNames->getNameForValue(ifOp.getCondition())
     << ") {\n";
  for (auto &op : thenBlock.getOperations()) {
    if (auto yieldOp = mlir::dyn_cast<scf::YieldOp>(op)) {
      os << "  " << resultName << " = std::move("
         << variableNames->getNameForValue(yieldOp->getOperand(0)) << ");\n";
    } else {
      os << "  ";
      if (failed(translate(op))) return failure();
    }
  }
  os << "} else {\n";
  for (auto &op : elseBlock.getOperations()) {
    if (auto yieldOp = mlir::dyn_cast<scf::YieldOp>(op)) {
      os << "  " << resultName << " = std::move("
         << variableNames->getNameForValue(yieldOp->getOperand(0)) << ");\n";
    } else {
      os << "  ";
      if (failed(translate(op))) return failure();
    }
  }
  os << "}\n";

  return success();
}

LogicalResult OpenFheBinEmitter::printOperation(affine::AffineForOp forOp) {
  auto inductionVar = variableNames->getNameForValue(forOp.getInductionVar());
  if (forOp.getNumRegionIterArgs() != 0) {
    forOp.emitError(
        "Loops with loop-carried dependence currently unsupported!");
    return failure();
  }
  os << "for (";
  if (failed(emitTypedAssignPrefix(forOp.getInductionVar(), forOp->getLoc()))) {
    return failure();
  };
  os << forOp.getConstantLowerBound() << "; ";
  os << inductionVar << " < " << forOp.getConstantUpperBound() << "; ";
  os << inductionVar << " += " << forOp.getStepAsInt() << ") {\n";
  for (auto &op : forOp.getBody()->getOperations()) {
    if (mlir::isa<affine::AffineYieldOp>(op)) continue;
    os << "  ";
    if (failed(translate(op))) {
      return failure();
    }
  }
  os << "}\n";
  return success();
}

LogicalResult OpenFheBinEmitter::printInPlaceEvalMethod(
    mlir::Value result, mlir::Value cryptoContext, mlir::ValueRange operands,
    std::string_view op, Location loc) {
  if (failed(emitTypedAssignPrefix(result, loc))) {
    return failure();
  }
  os << "copy(" << variableNames->getNameForValue(*operands.begin()) << ");\n";
  os << variableNames->getNameForValue(cryptoContext) << "->" << op << "("
     << variableNames->getNameForValue(result) << ", ";
  os << commaSeparatedValues(
      mlir::ValueRange(operands.begin() + 1, operands.end()),
      [&](mlir::Value value) { return variableNames->getNameForValue(value); });
  os << ");\n";
  return success();
}

void OpenFheBinEmitter::emitAutoAssignPrefix(Value result) {
  os << "auto ";
  os << variableNames->getNameForValue(result) << " = ";
}

LogicalResult OpenFheBinEmitter::emitTypedAssignPrefix(Value result,
                                                       Location loc,
                                                       bool constant) {
  auto typeResult = convertType(result.getType(), loc, constant);
  if (failed(typeResult)) {
    return failure();
  }
  os << typeResult.value() << " ";
  os << variableNames->getNameForValue(result) << " = ";
  return success();
}

LogicalResult OpenFheBinEmitter::emitType(Type type, Location loc,
                                          bool constant) {
  auto typeResult = convertType(type, loc, constant);
  if (failed(typeResult)) {
    return failure();
  }
  os << typeResult.value();
  return success();
}

LogicalResult OpenFheBinEmitter::printEvalMethod(
    ::mlir::Value result, ::mlir::Value cryptoContext,
    ::mlir::ValueRange nonEvalOperands, std::string_view op) {
  emitAutoAssignPrefix(result);
  os << variableNames->getNameForValue(cryptoContext) << "->" << op << "(";
  os << commaSeparatedValues(nonEvalOperands, [&](Value value) {
    return variableNames->getNameForValue(value);
  });
  os << ");\n";
  return success();
}

LogicalResult OpenFheBinEmitter::translateFallback(Operation &operation) {
  return emitError(operation.getLoc(),
                   "unable to find printer for op: " +
                       operation.getName().getStringRef().str());
}

}  // namespace mlir::heir::openfhe
