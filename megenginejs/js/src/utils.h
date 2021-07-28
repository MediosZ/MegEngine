#pragma once
#include "./webassembly.h"
#include "./grad.h"
#include "megbrain/imperative/ops/autogen.h"
#include <vector>
#include <unordered_map>
#include <string>

using namespace mgb::imperative::interpreter;
namespace mgb::imperative::js {

DType getDataType(int d);

HostTensorND makeGrad(const TensorShape& shape);

void printTensor(std::shared_ptr<Tensor> t);

std::shared_ptr<Tensor> make_shape(std::initializer_list<int32_t> init_shape);

#ifdef __EMSCRIPTEN__

void assignData(const emscripten::val &data, std::shared_ptr<HostTensorND> ret, const int type);

SmallVector<size_t> getVectorFromVal(const emscripten::val &v);

#endif 


}
