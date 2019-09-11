// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "assert.h"
namespace onnxruntime {
namespace contrib {

// Computes the squared Euclidean distance between the vectors.
template <typename T>
class Sqeuclidean {
 public:
  T operator()(const T* a1, const T* b1, size_t n) const {
    T sum = 0;
    for (size_t k = 0; k != n; ++k) {
      const T t = a1[k] - b1[k];
      sum += t * t;
    }
    return sum;
  }
};

// https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
//\param a: matrix with shape of[ma,n]
//\param b: matrix with shape of[mb,n]
//\param dest: matrix with shape of [ma,mb]
template <typename T, typename ElemFunc>
void cdist(const T* a, const T* b, T* dest, size_t ma, size_t mb, size_t n) {
  ElemFunc f;
  for (size_t i = 0; i != ma; ++i) {
    const T* a1 = a + n * i;
    for (size_t j = 0; j != mb; ++j) {
      const T* b1 = b + n * j;
      *dest++ = f(a1, b1, n);
    }
  }
}

template <typename T>
class CDist final : public OpKernel {
 private:
  typedef void (*DistFunc)(const T* a, const T* b, T* dest, size_t ma, size_t mb, size_t n);
  const DistFunc dist_func_;
  static DistFunc FindDistFunc(const OpKernelInfo& info) {
    std::string metric;
    ORT_ENFORCE(info.GetAttr<std::string>("metric", &metric).IsOK());
	if(metric.compare("sqeuclidean") == 0)
		return cdist<T, Sqeuclidean<T>>;
	return nullptr;
  }

 public:
  CDist(const OpKernelInfo& info) : OpKernel(info), dist_func_(FindDistFunc(info)) {
    if (dist_func_ == nullptr) ORT_NOT_IMPLEMENTED();
  }

  common::Status Compute(OpKernelContext* context) const override {
    assert(context->InputCount() == 2);
    const Tensor* A = context->Input<Tensor>(0);
    const Tensor* B = context->Input<Tensor>(1);
    const TensorShape& shape_a = A->Shape();
    const TensorShape& shape_b = B->Shape();
    if (shape_a.NumDimensions() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The first input of CDist kernel has wrong shape: ", shape_a);
    }

    if (shape_b.NumDimensions() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The second input of CDist kernel has wrong shape: ", shape_b);
    }
    if (shape_a[1] != shape_b[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input shape dimensions mismatch:", shape_a, " and ", shape_b);
    }

    TensorShape output_shape = {shape_a[0], shape_b[0]};
    Tensor* C = context->Output(0, output_shape);
    T* output = C->MutableData<T>();
    dist_func_(A->Data<T>(), B->Data<T>(), output, shape_a[0], shape_b[0], shape_a[1]);
    return Status::OK();
  }
};
}  // namespace contrib
}  // namespace onnxruntime