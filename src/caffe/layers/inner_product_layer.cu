#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }

  // *****************************************************************************
  // BANG: re-scaling top_diff based on L-2 norms of batch elements
  const Dtype beta = this->layer_param_.inner_product_param().beta();

  if (beta != 0.0) {
    const vector<bool>& classifications = Caffe::classifications();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* mutable_top_diff = top[0]->mutable_gpu_diff();

    const Dtype epsilon = this->layer_param_.inner_product_param().epsilon();
    const Dtype ratio = this->layer_param_.inner_product_param().ratio();

    // To store norms for elements of the batch
    int batch_size = top[0]->shape(0);
    vector<Dtype> l2_norms(batch_size);
    Dtype max_l2 = 0.0;
    int count = top[0]->count();
    int element_size = count / batch_size;

    for (int j = 0; j < batch_size; ++j) {
      Dtype diff_l2;
      caffe_gpu_dot(element_size, top_diff + j * element_size,
                    top_diff + j * element_size, &diff_l2);
      diff_l2 = std::sqrt(diff_l2);
      max_l2 = std::max(max_l2, diff_l2);
      l2_norms[j] = diff_l2;
    }

    // Scaling
    for (int j = 0; j < batch_size; ++j) {
      if (l2_norms[j] != 0.0) {
        Dtype eps = epsilon;
        if (classifications[j] == 0) {
          // Incorrectly classified sample
          eps = epsilon * ratio;
        }
        const Dtype scale = pow(max_l2 / l2_norms[j], eps * (1. - l2_norms[j] / max_l2));
        caffe_gpu_scal(element_size, scale, mutable_top_diff + j * element_size);
      }
    }
  }
  // *****************************************************************************

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }

  // *****************************************************************************
  // BANG: applying local learning rate (beta of BANG)
  if (beta != 0.0) {
    // Scale bias_diff with beta
    this->blobs_[1]->scale_diff(beta);
    // Scale weight_diff with beta
    this->blobs_[0]->scale_diff(beta);
  }
  // *****************************************************************************
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
