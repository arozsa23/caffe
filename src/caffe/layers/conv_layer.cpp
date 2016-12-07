#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

  // *****************************************************************************
  // BANG
  const Dtype beta = this->layer_param_.convolution_param().beta();
  const Dtype epsilon = this->layer_param_.convolution_param().epsilon();
  const Dtype ratio = this->layer_param_.convolution_param().ratio();
  const vector<bool>& classifications = Caffe::classifications();
  // *****************************************************************************

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    Dtype* mutable_top_diff = top[i]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

    // Bottom diff: Based on original top_diff
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                  bottom_diff + n * this->bottom_dim_);
        }
      }
    }

    // Weight diffs (including bias): Based on scaled original top_diff if necessary

    // *****************************************************************************
    // BANG: re-scaling top_diff based on L-2 norms of batch elements
    if (beta != 0.0) {
      int batch_size = top[i]->shape(0);
      // Storing L-2 norms for elements of the batch
      vector<Dtype> l2_norms(batch_size);
      Dtype max_l2 = 0.0;

      for (int j = 0; j < batch_size; ++j) {
        Dtype diff_l2 = std::sqrt(caffe_cpu_dot(this->top_dim_,top_diff + j * this->top_dim_,
                                  top_diff + j * this->top_dim_));
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
          const Dtype scale = pow( max_l2 / l2_norms[j], eps * (1. - l2_norms[j] / max_l2));
          caffe_scal(this->top_dim_, scale, mutable_top_diff + j * this->top_dim_);
        }
      }
    }
    // *****************************************************************************

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
      }
    }
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

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
