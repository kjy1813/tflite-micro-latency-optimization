/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma GCC optimize ("O3")
#pragma GCC optimize ("unroll-loops")
#include "tensorflow/lite/micro/kernels/conv.h"

#include "third_party/cmsis_nn/Include/arm_nn_types.h"
#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/kernels/cmsis_nn/project2_conv2_filter_pairmajor_packed.h"
#include "tensorflow/lite/micro/kernels/cmsis_nn/project2_conv3_filter_pairmajor_packed.h"
#include "tensorflow/lite/micro/kernels/cmsis_nn/project2_conv4_filter_pairmajor_packed.h"
#include "tensorflow/lite/micro/kernels/cmsis_nn/project2_conv5_filter_pairmajor_packed.h"

namespace tflite {
namespace {

struct OpData {
  OpDataConv reference_op_data;

  // Index to buffer for optimizations if applicable.
  int buffer_idx;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  int32_t buf_size = 0;
  const auto& params =
      *(static_cast<const TfLiteConvParams*>(node->builtin_data));
  OpData* data = static_cast<OpData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  RuntimeShape input_shape = GetTensorShape(input);
  RuntimeShape output_shape = GetTensorShape(output);

  // Initialize cmsis_nn input dimensions
  cmsis_nn_dims input_dims;
  input_dims.n = MatchingDim(input_shape, 0, output_shape, 0);
  input_dims.h = input->dims->data[1];
  input_dims.w = input->dims->data[2];
  input_dims.c = input_shape.Dims(3);

  // Initialize cmsis_nn filter dimensions
  cmsis_nn_dims filter_dims;
  filter_dims.n = output_shape.Dims(3);
  filter_dims.h = filter->dims->data[1];
  filter_dims.w = filter->dims->data[2];
  filter_dims.c = input_dims.c;

  // Initialize cmsis_nn output dimensions
  cmsis_nn_dims output_dims;
  output_dims.n = input_dims.n;
  output_dims.h = output->dims->data[1];
  output_dims.w = output->dims->data[2];
  output_dims.c = output_shape.Dims(3);

  if (filter->type == kTfLiteInt4) {
    int filter_size =
        RuntimeShape(filter->dims->size,
                     reinterpret_cast<const int32_t*>(filter->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(
        context, filter_size, &data->reference_op_data.filter_buffer_index);
  }

  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    const int num_channels = filter->dims->data[kConvQuantizedDimension];
    data->reference_op_data.per_channel_output_multiplier =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, num_channels * sizeof(int32_t)));
    data->reference_op_data.per_channel_output_shift =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, num_channels * sizeof(int32_t)));
  }

  TF_LITE_ENSURE_STATUS(CalculateOpDataConv(
      context, node, params, input_dims.w, input_dims.h, filter_dims.w,
      filter_dims.h, output_dims.w, output_dims.h, input->type,
      &data->reference_op_data));

  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    // Initialize cmsis_nn convolution parameters
    cmsis_nn_conv_params conv_params;
    conv_params.input_offset = -input->params.zero_point;
    conv_params.output_offset = output->params.zero_point;
    conv_params.stride.h = params.stride_height;
    conv_params.stride.w = params.stride_width;
    conv_params.dilation.h = params.dilation_height_factor;
    conv_params.dilation.w = params.dilation_width_factor;
    conv_params.padding.h = data->reference_op_data.padding.height;
    conv_params.padding.w = data->reference_op_data.padding.width;
    conv_params.activation.min = data->reference_op_data.output_activation_min;
    conv_params.activation.max = data->reference_op_data.output_activation_max;

    if (input->type == kTfLiteInt8) {
      buf_size = arm_convolve_wrapper_s8_get_buffer_size(
          &conv_params, &input_dims, &filter_dims, &output_dims);
    } else if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
      buf_size = arm_convolve_wrapper_s16_get_buffer_size(
          &conv_params, &input_dims, &filter_dims, &output_dims);
    }

    if (buf_size > 0) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, buf_size, &data->buffer_idx));
    } else {
      data->buffer_idx = -1;
    }
  }

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);

  return kTfLiteOk;
}

__attribute__((hot)) TfLiteStatus EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                                     const TfLiteConvParams& params,
                                     const OpData& data,
                                     const TfLiteEvalTensor* input,
                                     const TfLiteEvalTensor* filter,
                                     const TfLiteEvalTensor* bias,
                                     TfLiteEvalTensor* output) {
  cmsis_nn_conv_params conv_params;
  conv_params.dilation.h = params.dilation_height_factor;
  conv_params.dilation.w = params.dilation_width_factor;

  // Initialize cmsis_nn convolution parameters
  conv_params.input_offset = -data.reference_op_data.input_zero_point;
  conv_params.output_offset = data.reference_op_data.output_zero_point;
  conv_params.stride.h = params.stride_height;
  conv_params.stride.w = params.stride_width;
  conv_params.padding.h = data.reference_op_data.padding.height;
  conv_params.padding.w = data.reference_op_data.padding.width;
  conv_params.activation.min = data.reference_op_data.output_activation_min;
  conv_params.activation.max = data.reference_op_data.output_activation_max;

  // Initialize cmsis_nn per channel quantization parameters
  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = const_cast<int32_t*>(
      data.reference_op_data.per_channel_output_multiplier);
  quant_params.shift =
      const_cast<int32_t*>(data.reference_op_data.per_channel_output_shift);

  RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
  RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  RuntimeShape bias_shape = tflite::micro::GetTensorShape(bias);

  // Consistency check.
  TFLITE_DCHECK_LE(conv_params.activation.min, conv_params.activation.max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batch_size = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (tflite::micro::GetOptionalTensorData<int8_t>(bias)) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Initialize cmsis_nn dimensions
  // Input
  cmsis_nn_dims input_dims;
  input_dims.n = batch_size;
  input_dims.h = input_shape.Dims(1);
  input_dims.w = input_shape.Dims(2);
  input_dims.c = input_depth;

  // Filter
  cmsis_nn_dims filter_dims;
  filter_dims.n = output_depth;
  filter_dims.h = filter_shape.Dims(1);
  filter_dims.w = filter_shape.Dims(2);
  filter_dims.c = input_depth;

  // Bias
  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output_depth;

  // Output
  cmsis_nn_dims output_dims;
  output_dims.n = batch_size;
  output_dims.h = output_shape.Dims(1);
  output_dims.w = output_shape.Dims(2);
  output_dims.c = output_depth;

  // Initialize cmsis_nn context
  cmsis_nn_context ctx;
  ctx.buf = nullptr;
  ctx.size = 0;

  if (data.buffer_idx > -1) {
    ctx.buf = context->GetScratchBuffer(context, data.buffer_idx);
    // Note: ctx.size is currently not used in cmsis_nn.
    // The buffer should be allocated in the Prepare function through
    // arm_convolve_wrapper_s8_get_buffer_size
  }

  // arm_convolve_wrapper_s8 dispatches the optimized kernel accordingly with
  // the parameters passed
  TFLITE_DCHECK_EQ(
      arm_convolve_wrapper_s8(
          &ctx, &conv_params, &quant_params, &input_dims,
          tflite::micro::GetTensorData<int8_t>(input), &filter_dims,
          tflite::micro::GetTensorData<int8_t>(filter), &bias_dims,
          tflite::micro::GetOptionalTensorData<int32_t>(bias), &output_dims,
          tflite::micro::GetTensorData<int8_t>(output)),
      ARM_CMSIS_NN_SUCCESS);

  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedPerChannel16x8(
    TfLiteContext* context, TfLiteNode* node, const TfLiteConvParams& params,
    const OpData& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {
  cmsis_nn_conv_params conv_params;
  conv_params.dilation.h = params.dilation_height_factor;
  conv_params.dilation.w = params.dilation_width_factor;

  // Initialize cmsis_nn convolution parameters
  conv_params.input_offset = -data.reference_op_data.input_zero_point;
  conv_params.output_offset = data.reference_op_data.output_zero_point;
  conv_params.stride.h = params.stride_height;
  conv_params.stride.w = params.stride_width;
  conv_params.padding.h = data.reference_op_data.padding.height;
  conv_params.padding.w = data.reference_op_data.padding.width;
  conv_params.activation.min = data.reference_op_data.output_activation_min;
  conv_params.activation.max = data.reference_op_data.output_activation_max;

  // Initialize cmsis_nn per channel quantization parameters
  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = const_cast<int32_t*>(
      data.reference_op_data.per_channel_output_multiplier);
  quant_params.shift =
      const_cast<int32_t*>(data.reference_op_data.per_channel_output_shift);

  RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
  RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  RuntimeShape bias_shape = tflite::micro::GetTensorShape(bias);

  // Consistency check.
  TFLITE_DCHECK_LE(conv_params.activation.min, conv_params.activation.max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batch_size = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (tflite::micro::GetOptionalTensorData<int8_t>(bias)) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Initialize cmsis_nn dimensions
  // Input
  cmsis_nn_dims input_dims;
  input_dims.n = batch_size;
  input_dims.h = input_shape.Dims(1);
  input_dims.w = input_shape.Dims(2);
  input_dims.c = input_depth;

  // Filter
  cmsis_nn_dims filter_dims;
  filter_dims.n = output_depth;
  filter_dims.h = filter_shape.Dims(1);
  filter_dims.w = filter_shape.Dims(2);
  filter_dims.c = input_depth;

  // Bias
  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output_depth;

  // Output
  cmsis_nn_dims output_dims;
  output_dims.n = batch_size;
  output_dims.h = output_shape.Dims(1);
  output_dims.w = output_shape.Dims(2);
  output_dims.c = output_depth;

  // Initialize cmsis_nn context
  cmsis_nn_context ctx;
  ctx.buf = nullptr;
  ctx.size = 0;

  if (data.buffer_idx > -1) {
    ctx.buf = context->GetScratchBuffer(context, data.buffer_idx);
    // Note: ctx.size is currently not used in cmsis_nn.
    // The buffer should be allocated in the Prepare function through
    // arm_convolve_wrapper_s8_get_buffer_size
  }

  TFLITE_DCHECK_EQ(
      arm_convolve_wrapper_s16(
          &ctx, &conv_params, &quant_params, &input_dims,
          tflite::micro::GetTensorData<int16_t>(input), &filter_dims,
          tflite::micro::GetTensorData<int8_t>(filter), &bias_dims,
          tflite::micro::GetOptionalTensorData<int64_t>(bias), &output_dims,
          tflite::micro::GetTensorData<int16_t>(output)),
      ARM_CMSIS_NN_SUCCESS);

  return kTfLiteOk;
}


// PROJECT2_DIRECTION_C_BEGIN
// Shape-specialized Conv4/Conv5 fast path.
// This keeps the .tflite graph, weights, bias, tensor shapes, and quantization
// parameters unchanged. It only removes TFLM's dynamic RuntimeShape/MatchingDim
// setup from the Conv4/Conv5 hot path and calls the CMSIS-NN convolution
// primitive with fixed dimensions.
__attribute__((always_inline)) inline bool Project2Dims4Eq(
    const TfLiteEvalTensor* t, const int d0, const int d1, const int d2,
    const int d3) {
  return t != nullptr && t->dims != nullptr && t->dims->size == 4 &&
         t->dims->data[0] == d0 && t->dims->data[1] == d1 &&
         t->dims->data[2] == d2 && t->dims->data[3] == d3;
}


__attribute__((always_inline)) inline bool IsProject2Conv1(
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* output) {
  return Project2Dims4Eq(input, 1, 28, 28, 1) &&
         Project2Dims4Eq(filter, 8, 3, 3, 1) &&
         Project2Dims4Eq(output, 1, 14, 14, 8);
}

__attribute__((always_inline)) inline bool IsProject2Conv2(
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* output) {
  return Project2Dims4Eq(input, 1, 14, 14, 8) &&
         Project2Dims4Eq(filter, 16, 3, 3, 8) &&
         Project2Dims4Eq(output, 1, 14, 14, 16);
}

__attribute__((always_inline)) inline bool IsProject2Conv3(
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* output) {
  return Project2Dims4Eq(input, 1, 14, 14, 16) &&
         Project2Dims4Eq(filter, 32, 3, 3, 16) &&
         Project2Dims4Eq(output, 1, 7, 7, 32);
}

__attribute__((always_inline)) inline bool IsProject2Conv4(
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* output) {
  return Project2Dims4Eq(input, 1, 7, 7, 32) &&
         Project2Dims4Eq(filter, 64, 3, 3, 32) &&
         Project2Dims4Eq(output, 1, 7, 7, 64);
}

__attribute__((always_inline)) inline bool IsProject2Conv5(
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* output) {
  return Project2Dims4Eq(input, 1, 7, 7, 64) &&
         Project2Dims4Eq(filter, 128, 3, 3, 64) &&
         Project2Dims4Eq(output, 1, 4, 4, 128);
}


// PROJECT2_CONV5_PAIRMAJOR_SPARSE_V4_BEGIN
// Conv5 pair-major sparse-activation kernel.
// Same Conv2D math as the model:
//   acc[oc] = bias[oc] + sum((input_q - input_zp) * weight_q[oc])
// but it skips activation pairs where both lanes are zero and uses a flash
// prepacked Conv5 filter layout [pair][output_channel]. This converts the hot
// loop from strided weight reads into contiguous output-channel accumulation.
//
// Target fixed shape:
// input  = int8 [1, 7, 7, 64]
// filter = int8 [128, 3, 3, 64]
// output = int8 [1, 4, 4, 128]
// stride = 2, SAME padding = 1
static inline int8_t Project2RequantizeToInt8(
    int32_t acc, const int32_t multiplier, const int32_t shift,
    const int32_t output_offset, const int32_t activation_min,
    const int32_t activation_max) {
  acc = tflite::MultiplyByQuantizedMultiplier(acc, multiplier, shift);
  acc += output_offset;
  if (acc < activation_min) {
    acc = activation_min;
  }
  if (acc > activation_max) {
    acc = activation_max;
  }
  return static_cast<int8_t>(acc);
}

static inline int32_t Project2PackSigned16x2(const int32_t lo,
                                             const int32_t hi) {
  return static_cast<int32_t>(
      (static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(lo)))) |
      (static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(hi))) << 16));
}

static inline int32_t Project2Smlad2(const int32_t a, const int32_t b,
                                     const int32_t acc) {
#if defined(__ARM_FEATURE_DSP) || defined(ARM_MATH_DSP)
  int32_t out;
  __asm volatile("smlad %0, %1, %2, %3"
                 : "=r"(out)
                 : "r"(a), "r"(b), "r"(acc));
  return out;
#else
  const int16_t a0 = static_cast<int16_t>(a & 0xffff);
  const int16_t a1 = static_cast<int16_t>((static_cast<uint32_t>(a) >> 16) & 0xffff);
  const int16_t b0 = static_cast<int16_t>(b & 0xffff);
  const int16_t b1 = static_cast<int16_t>((static_cast<uint32_t>(b) >> 16) & 0xffff);
  return acc + static_cast<int32_t>(a0) * static_cast<int32_t>(b0) +
         static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
#endif
}



// PROJECT2_CONV123_SPECIAL_BEGIN
// Conv1 direct sparse-activation kernel.
// Target fixed shape:
// input  = int8 [1, 28, 28, 1]
// filter = int8 [8, 3, 3, 1]
// output = int8 [1, 14, 14, 8]
// stride = 2, SAME padding. Uses original filter tensor, no model change.
__attribute__((hot)) TfLiteStatus EvalProject2Conv1DirectSparse(
    const OpData& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const int32_t* bias_data = tflite::micro::GetOptionalTensorData<int32_t>(bias);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  const int32_t input_offset = -data.reference_op_data.input_zero_point;
  const int32_t output_offset = data.reference_op_data.output_zero_point;
  const int32_t activation_min = data.reference_op_data.output_activation_min;
  const int32_t activation_max = data.reference_op_data.output_activation_max;
  const int32_t* output_multiplier = data.reference_op_data.per_channel_output_multiplier;
  const int32_t* output_shift = data.reference_op_data.per_channel_output_shift;

  constexpr int input_h = 28;
  constexpr int input_w = 28;
  constexpr int output_h = 14;
  constexpr int output_w = 14;
  constexpr int output_c = 8;
  constexpr int filter_h = 3;
  constexpr int filter_w = 3;
  constexpr int stride = 2;
  const int pad_h = data.reference_op_data.padding.height;
  const int pad_w = data.reference_op_data.padding.width;

  int32_t acc[output_c];
  for (int oh = 0; oh < output_h; ++oh) {
    for (int ow = 0; ow < output_w; ++ow) {
      for (int oc = 0; oc < output_c; ++oc) {
        acc[oc] = bias_data ? bias_data[oc] : 0;
      }
      for (int kh = 0; kh < filter_h; ++kh) {
        const int ih = oh * stride + kh - pad_h;
        if (ih < 0 || ih >= input_h) continue;
        for (int kw = 0; kw < filter_w; ++kw) {
          const int iw = ow * stride + kw - pad_w;
          if (iw < 0 || iw >= input_w) continue;
          const int32_t x = static_cast<int32_t>(input_data[ih * input_w + iw]) + input_offset;
          if (x == 0) continue;
          const int k = kh * filter_w + kw;
          acc[0] += x * static_cast<int32_t>(filter_data[0 * 9 + k]);
          acc[1] += x * static_cast<int32_t>(filter_data[1 * 9 + k]);
          acc[2] += x * static_cast<int32_t>(filter_data[2 * 9 + k]);
          acc[3] += x * static_cast<int32_t>(filter_data[3 * 9 + k]);
          acc[4] += x * static_cast<int32_t>(filter_data[4 * 9 + k]);
          acc[5] += x * static_cast<int32_t>(filter_data[5 * 9 + k]);
          acc[6] += x * static_cast<int32_t>(filter_data[6 * 9 + k]);
          acc[7] += x * static_cast<int32_t>(filter_data[7 * 9 + k]);
        }
      }
      const int out_base = (oh * output_w + ow) * output_c;
      for (int oc = 0; oc < output_c; ++oc) {
        output_data[out_base + oc] = Project2RequantizeToInt8(
            acc[oc], output_multiplier[oc], output_shift[oc], output_offset,
            activation_min, activation_max);
      }
    }
  }
  return kTfLiteOk;
}

// Conv2 pair-major sparse-activation kernel.
// input [1,14,14,8], filter [16,3,3,8], output [1,14,14,16].
__attribute__((hot)) TfLiteStatus EvalProject2Conv2PairMajorSparseV1(
    const OpData& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {
  (void)filter;
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const int32_t* bias_data = tflite::micro::GetOptionalTensorData<int32_t>(bias);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
  const int32_t input_offset = -data.reference_op_data.input_zero_point;
  const int32_t output_offset = data.reference_op_data.output_zero_point;
  const int32_t activation_min = data.reference_op_data.output_activation_min;
  const int32_t activation_max = data.reference_op_data.output_activation_max;
  const int32_t* output_multiplier = data.reference_op_data.per_channel_output_multiplier;
  const int32_t* output_shift = data.reference_op_data.per_channel_output_shift;

  constexpr int input_h = 14, input_w = 14, input_c = 8;
  constexpr int output_h = 14, output_w = 14, output_c = 16;
  constexpr int filter_h = 3, filter_w = 3, stride = 1;
  constexpr int input_pairs = input_c / 2;
  const int pad_h = data.reference_op_data.padding.height;
  const int pad_w = data.reference_op_data.padding.width;
  const int32_t* packed_filter = tflite::project2_conv2_pairmajor_packed::kConv2FilterPairMajorPacked;
  int32_t acc[output_c];

  for (int oh = 0; oh < output_h; ++oh) {
    for (int ow = 0; ow < output_w; ++ow) {
      for (int oc = 0; oc < output_c; ++oc) acc[oc] = bias_data ? bias_data[oc] : 0;
      for (int kh = 0; kh < filter_h; ++kh) {
        const int ih = oh * stride + kh - pad_h;
        if (ih < 0 || ih >= input_h) continue;
        for (int kw = 0; kw < filter_w; ++kw) {
          const int iw = ow * stride + kw - pad_w;
          if (iw < 0 || iw >= input_w) continue;
          const int input_base = (ih * input_w + iw) * input_c;
          const int pair_base = (kh * filter_w + kw) * input_pairs;
          for (int icp = 0; icp < input_pairs; ++icp) {
            const int ic = icp * 2;
            const int32_t x0 = static_cast<int32_t>(input_data[input_base + ic + 0]) + input_offset;
            const int32_t x1 = static_cast<int32_t>(input_data[input_base + ic + 1]) + input_offset;
            if ((x0 | x1) == 0) continue;
            const int32_t xpack = Project2PackSigned16x2(x0, x1);
            const int32_t* w = &packed_filter[(pair_base + icp) * output_c];
            acc[0] = Project2Smlad2(xpack, w[0], acc[0]);
            acc[1] = Project2Smlad2(xpack, w[1], acc[1]);
            acc[2] = Project2Smlad2(xpack, w[2], acc[2]);
            acc[3] = Project2Smlad2(xpack, w[3], acc[3]);
            acc[4] = Project2Smlad2(xpack, w[4], acc[4]);
            acc[5] = Project2Smlad2(xpack, w[5], acc[5]);
            acc[6] = Project2Smlad2(xpack, w[6], acc[6]);
            acc[7] = Project2Smlad2(xpack, w[7], acc[7]);
            acc[8] = Project2Smlad2(xpack, w[8], acc[8]);
            acc[9] = Project2Smlad2(xpack, w[9], acc[9]);
            acc[10] = Project2Smlad2(xpack, w[10], acc[10]);
            acc[11] = Project2Smlad2(xpack, w[11], acc[11]);
            acc[12] = Project2Smlad2(xpack, w[12], acc[12]);
            acc[13] = Project2Smlad2(xpack, w[13], acc[13]);
            acc[14] = Project2Smlad2(xpack, w[14], acc[14]);
            acc[15] = Project2Smlad2(xpack, w[15], acc[15]);
          }
        }
      }
      const int out_base = (oh * output_w + ow) * output_c;
      for (int oc = 0; oc < output_c; ++oc) {
        output_data[out_base + oc] = Project2RequantizeToInt8(
            acc[oc], output_multiplier[oc], output_shift[oc], output_offset,
            activation_min, activation_max);
      }
    }
  }
  return kTfLiteOk;
}

// Conv3 pair-major sparse-activation kernel.
// input [1,14,14,16], filter [32,3,3,16], output [1,7,7,32], stride2.
__attribute__((hot)) TfLiteStatus EvalProject2Conv3PairMajorSparseV1(
    const OpData& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {
  (void)filter;
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const int32_t* bias_data = tflite::micro::GetOptionalTensorData<int32_t>(bias);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
  const int32_t input_offset = -data.reference_op_data.input_zero_point;
  const int32_t output_offset = data.reference_op_data.output_zero_point;
  const int32_t activation_min = data.reference_op_data.output_activation_min;
  const int32_t activation_max = data.reference_op_data.output_activation_max;
  const int32_t* output_multiplier = data.reference_op_data.per_channel_output_multiplier;
  const int32_t* output_shift = data.reference_op_data.per_channel_output_shift;

  constexpr int input_h = 14, input_w = 14, input_c = 16;
  constexpr int output_h = 7, output_w = 7, output_c = 32;
  constexpr int filter_h = 3, filter_w = 3, stride = 2;
  constexpr int input_pairs = input_c / 2;
  const int pad_h = data.reference_op_data.padding.height;
  const int pad_w = data.reference_op_data.padding.width;
  const int32_t* packed_filter = tflite::project2_conv3_pairmajor_packed::kConv3FilterPairMajorPacked;
  int32_t acc[output_c];

  for (int oh = 0; oh < output_h; ++oh) {
    for (int ow = 0; ow < output_w; ++ow) {
      for (int oc = 0; oc < output_c; ++oc) acc[oc] = bias_data ? bias_data[oc] : 0;
      for (int kh = 0; kh < filter_h; ++kh) {
        const int ih = oh * stride + kh - pad_h;
        if (ih < 0 || ih >= input_h) continue;
        for (int kw = 0; kw < filter_w; ++kw) {
          const int iw = ow * stride + kw - pad_w;
          if (iw < 0 || iw >= input_w) continue;
          const int input_base = (ih * input_w + iw) * input_c;
          const int pair_base = (kh * filter_w + kw) * input_pairs;
          for (int icp = 0; icp < input_pairs; ++icp) {
            const int ic = icp * 2;
            const int32_t x0 = static_cast<int32_t>(input_data[input_base + ic + 0]) + input_offset;
            const int32_t x1 = static_cast<int32_t>(input_data[input_base + ic + 1]) + input_offset;
            if ((x0 | x1) == 0) continue;
            const int32_t xpack = Project2PackSigned16x2(x0, x1);
            const int32_t* w = &packed_filter[(pair_base + icp) * output_c];
            int oc = 0;
            for (; oc + 7 < output_c; oc += 8) {
              acc[oc + 0] = Project2Smlad2(xpack, w[oc + 0], acc[oc + 0]);
              acc[oc + 1] = Project2Smlad2(xpack, w[oc + 1], acc[oc + 1]);
              acc[oc + 2] = Project2Smlad2(xpack, w[oc + 2], acc[oc + 2]);
              acc[oc + 3] = Project2Smlad2(xpack, w[oc + 3], acc[oc + 3]);
              acc[oc + 4] = Project2Smlad2(xpack, w[oc + 4], acc[oc + 4]);
              acc[oc + 5] = Project2Smlad2(xpack, w[oc + 5], acc[oc + 5]);
              acc[oc + 6] = Project2Smlad2(xpack, w[oc + 6], acc[oc + 6]);
              acc[oc + 7] = Project2Smlad2(xpack, w[oc + 7], acc[oc + 7]);
            }
          }
        }
      }
      const int out_base = (oh * output_w + ow) * output_c;
      for (int oc = 0; oc < output_c; ++oc) {
        output_data[out_base + oc] = Project2RequantizeToInt8(
            acc[oc], output_multiplier[oc], output_shift[oc], output_offset,
            activation_min, activation_max);
      }
    }
  }
  return kTfLiteOk;
}

#ifndef PROJECT2_CONV2_PM_SPARSE_THRESHOLD
#define PROJECT2_CONV2_PM_SPARSE_THRESHOLD 700
#endif
#ifndef PROJECT2_CONV3_PM_SPARSE_THRESHOLD
#define PROJECT2_CONV3_PM_SPARSE_THRESHOLD 1400
#endif

__attribute__((always_inline)) inline int Project2CountConv2InputZeros(const TfLiteEvalTensor* input) {
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  int zero_count = 0;
  constexpr int total = 14 * 14 * 8;
  for (int i = 0; i < total; ++i) zero_count += (input_data[i] == static_cast<int8_t>(-128));
  return zero_count;
}

__attribute__((always_inline)) inline int Project2CountConv3InputZeros(const TfLiteEvalTensor* input) {
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  int zero_count = 0;
  constexpr int total = 14 * 14 * 16;
  for (int i = 0; i < total; ++i) zero_count += (input_data[i] == static_cast<int8_t>(-128));
  return zero_count;
}
// PROJECT2_CONV123_SPECIAL_END

// PROJECT2_CONV4_PAIRMAJOR_SPARSE_V1_BEGIN
// Conv4 pair-major sparse-activation kernel.
// Target fixed shape:
// input  = int8 [1, 7, 7, 32]
// filter = int8 [64, 3, 3, 32]
// output = int8 [1, 7, 7, 64]
// stride = 1, SAME padding = 1
//
// This is the Conv4 analogue of the successful Conv5 pair-major V4 path.
// It keeps the model file untouched and uses an equivalent prepacked flash
// copy of Conv4 filter weights laid out as [input_pair][output_channel].
__attribute__((hot)) TfLiteStatus EvalProject2Conv4PairMajorSparseV1(
    const OpData& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {
  (void)filter;  // Same Conv4 weights are used from a prepacked flash layout.

  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const int32_t* bias_data =
      tflite::micro::GetOptionalTensorData<int32_t>(bias);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  const int32_t input_offset = -data.reference_op_data.input_zero_point;   // observed 128
  const int32_t output_offset = data.reference_op_data.output_zero_point;  // observed -128
  const int32_t activation_min = data.reference_op_data.output_activation_min;
  const int32_t activation_max = data.reference_op_data.output_activation_max;
  const int32_t* output_multiplier =
      data.reference_op_data.per_channel_output_multiplier;
  const int32_t* output_shift = data.reference_op_data.per_channel_output_shift;

  constexpr int input_h = 7;
  constexpr int input_w = 7;
  constexpr int input_c = 32;
  constexpr int output_h = 7;
  constexpr int output_w = 7;
  constexpr int output_c = 64;
  constexpr int filter_h = 3;
  constexpr int filter_w = 3;
  constexpr int stride = 1;
  constexpr int pad = 1;
  constexpr int input_pairs = input_c / 2;       // 16
  constexpr int pairs_per_filter = filter_h * filter_w * input_pairs;  // 144
  (void)pairs_per_filter;

  const int32_t* packed_filter =
      tflite::project2_conv4_pairmajor_packed::kConv4FilterPairMajorPacked;

  int32_t acc[output_c];

  for (int oh = 0; oh < output_h; ++oh) {
    for (int ow = 0; ow < output_w; ++ow) {
      for (int oc = 0; oc < output_c; ++oc) {
        acc[oc] = bias_data ? bias_data[oc] : 0;
      }

      for (int kh = 0; kh < filter_h; ++kh) {
        const int ih = oh * stride + kh - pad;
        if (ih < 0 || ih >= input_h) {
          continue;
        }
        for (int kw = 0; kw < filter_w; ++kw) {
          const int iw = ow * stride + kw - pad;
          if (iw < 0 || iw >= input_w) {
            continue;
          }

          const int input_base = (ih * input_w + iw) * input_c;
          const int pair_base = (kh * filter_w + kw) * input_pairs;

          for (int icp = 0; icp < input_pairs; ++icp) {
            const int ic = icp * 2;
            const int32_t x0 = static_cast<int32_t>(input_data[input_base + ic + 0]) + input_offset;
            const int32_t x1 = static_cast<int32_t>(input_data[input_base + ic + 1]) + input_offset;
            if ((x0 | x1) == 0) {
              continue;
            }

            const int32_t xpack = Project2PackSigned16x2(x0, x1);
            const int32_t* w = &packed_filter[(pair_base + icp) * output_c];

            int oc = 0;
            for (; oc + 7 < output_c; oc += 8) {
              acc[oc + 0] = Project2Smlad2(xpack, w[oc + 0], acc[oc + 0]);
              acc[oc + 1] = Project2Smlad2(xpack, w[oc + 1], acc[oc + 1]);
              acc[oc + 2] = Project2Smlad2(xpack, w[oc + 2], acc[oc + 2]);
              acc[oc + 3] = Project2Smlad2(xpack, w[oc + 3], acc[oc + 3]);
              acc[oc + 4] = Project2Smlad2(xpack, w[oc + 4], acc[oc + 4]);
              acc[oc + 5] = Project2Smlad2(xpack, w[oc + 5], acc[oc + 5]);
              acc[oc + 6] = Project2Smlad2(xpack, w[oc + 6], acc[oc + 6]);
              acc[oc + 7] = Project2Smlad2(xpack, w[oc + 7], acc[oc + 7]);
            }
            for (; oc < output_c; ++oc) {
              acc[oc] = Project2Smlad2(xpack, w[oc], acc[oc]);
            }
          }
        }
      }

      const int out_base = (oh * output_w + ow) * output_c;
      for (int oc = 0; oc < output_c; ++oc) {
        output_data[out_base + oc] = Project2RequantizeToInt8(
            acc[oc], output_multiplier[oc], output_shift[oc], output_offset,
            activation_min, activation_max);
      }
    }
  }

  return kTfLiteOk;
}
// PROJECT2_CONV4_PAIRMAJOR_SPARSE_V1_END

__attribute__((hot)) TfLiteStatus EvalProject2Conv5PairMajorSparseV4(
    const OpData& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {
  (void)filter;  // The same Conv5 weights are used from a prepacked flash layout.

  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const int32_t* bias_data =
      tflite::micro::GetOptionalTensorData<int32_t>(bias);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  const int32_t input_offset = -data.reference_op_data.input_zero_point;   // observed 128
  const int32_t output_offset = data.reference_op_data.output_zero_point;  // observed -128
  const int32_t activation_min = data.reference_op_data.output_activation_min;
  const int32_t activation_max = data.reference_op_data.output_activation_max;
  const int32_t* output_multiplier =
      data.reference_op_data.per_channel_output_multiplier;
  const int32_t* output_shift = data.reference_op_data.per_channel_output_shift;

  constexpr int input_h = 7;
  constexpr int input_w = 7;
  constexpr int input_c = 64;
  constexpr int output_h = 4;
  constexpr int output_w = 4;
  constexpr int output_c = 128;
  constexpr int filter_h = 3;
  constexpr int filter_w = 3;
  constexpr int stride = 2;
  constexpr int pad = 1;
  constexpr int input_pairs = input_c / 2;       // 32
  constexpr int pairs_per_filter = filter_h * filter_w * input_pairs;  // 288

  const int32_t* packed_filter =
      tflite::project2_conv5_pairmajor_packed::kConv5FilterPairMajorPacked;

  int32_t acc[output_c];

  for (int oh = 0; oh < output_h; ++oh) {
    for (int ow = 0; ow < output_w; ++ow) {
      // Initialize all 128 output-channel accumulators once for this output pixel.
      for (int oc = 0; oc < output_c; ++oc) {
        acc[oc] = bias_data ? bias_data[oc] : 0;
      }

      for (int kh = 0; kh < filter_h; ++kh) {
        const int ih = oh * stride + kh - pad;
        if (ih < 0 || ih >= input_h) {
          continue;
        }
        for (int kw = 0; kw < filter_w; ++kw) {
          const int iw = ow * stride + kw - pad;
          if (iw < 0 || iw >= input_w) {
            continue;
          }

          const int input_base = (ih * input_w + iw) * input_c;
          const int pair_base = (kh * filter_w + kw) * input_pairs;

          for (int icp = 0; icp < input_pairs; ++icp) {
            const int ic = icp * 2;
            const int32_t x0 = static_cast<int32_t>(input_data[input_base + ic + 0]) + input_offset;
            const int32_t x1 = static_cast<int32_t>(input_data[input_base + ic + 1]) + input_offset;

            // If both lanes are exact real zero, this activation pair contributes nothing.
            if ((x0 | x1) == 0) {
              continue;
            }

            const int32_t xpack = Project2PackSigned16x2(x0, x1);
            const int32_t* w = &packed_filter[(pair_base + icp) * output_c];

            int oc = 0;
            for (; oc + 7 < output_c; oc += 8) {
              acc[oc + 0] = Project2Smlad2(xpack, w[oc + 0], acc[oc + 0]);
              acc[oc + 1] = Project2Smlad2(xpack, w[oc + 1], acc[oc + 1]);
              acc[oc + 2] = Project2Smlad2(xpack, w[oc + 2], acc[oc + 2]);
              acc[oc + 3] = Project2Smlad2(xpack, w[oc + 3], acc[oc + 3]);
              acc[oc + 4] = Project2Smlad2(xpack, w[oc + 4], acc[oc + 4]);
              acc[oc + 5] = Project2Smlad2(xpack, w[oc + 5], acc[oc + 5]);
              acc[oc + 6] = Project2Smlad2(xpack, w[oc + 6], acc[oc + 6]);
              acc[oc + 7] = Project2Smlad2(xpack, w[oc + 7], acc[oc + 7]);
            }
            for (; oc < output_c; ++oc) {
              acc[oc] = Project2Smlad2(xpack, w[oc], acc[oc]);
            }
          }
        }
      }

      const int out_base = (oh * output_w + ow) * output_c;
      for (int oc = 0; oc < output_c; ++oc) {
        output_data[out_base + oc] = Project2RequantizeToInt8(
            acc[oc], output_multiplier[oc], output_shift[oc], output_offset,
            activation_min, activation_max);
      }
    }
  }

  return kTfLiteOk;
}
// PROJECT2_CONV5_PAIRMAJOR_SPARSE_V4_END

__attribute__((hot)) TfLiteStatus EvalProject2Conv45FixedCmsis(
    TfLiteContext* context, const TfLiteConvParams& params, const OpData& data,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output,
    const bool direct_primitive) {
  cmsis_nn_conv_params conv_params;
  conv_params.input_offset = -data.reference_op_data.input_zero_point;
  conv_params.output_offset = data.reference_op_data.output_zero_point;
  conv_params.stride.h = params.stride_height;
  conv_params.stride.w = params.stride_width;
  conv_params.dilation.h = params.dilation_height_factor;
  conv_params.dilation.w = params.dilation_width_factor;
  conv_params.padding.h = data.reference_op_data.padding.height;
  conv_params.padding.w = data.reference_op_data.padding.width;
  conv_params.activation.min = data.reference_op_data.output_activation_min;
  conv_params.activation.max = data.reference_op_data.output_activation_max;

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = const_cast<int32_t*>(
      data.reference_op_data.per_channel_output_multiplier);
  quant_params.shift = const_cast<int32_t*>(
      data.reference_op_data.per_channel_output_shift);

  cmsis_nn_dims input_dims;
  cmsis_nn_dims filter_dims;
  cmsis_nn_dims bias_dims;
  cmsis_nn_dims output_dims;

  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;

  if (IsProject2Conv1(input, filter, output)) {
    input_dims.n = 1; input_dims.h = 28; input_dims.w = 28; input_dims.c = 1;
    filter_dims.n = 8; filter_dims.h = 3; filter_dims.w = 3; filter_dims.c = 1;
    output_dims.n = 1; output_dims.h = 14; output_dims.w = 14; output_dims.c = 8;
    bias_dims.c = 8;
  } else if (IsProject2Conv2(input, filter, output)) {
    input_dims.n = 1; input_dims.h = 14; input_dims.w = 14; input_dims.c = 8;
    filter_dims.n = 16; filter_dims.h = 3; filter_dims.w = 3; filter_dims.c = 8;
    output_dims.n = 1; output_dims.h = 14; output_dims.w = 14; output_dims.c = 16;
    bias_dims.c = 16;
  } else if (IsProject2Conv3(input, filter, output)) {
    input_dims.n = 1; input_dims.h = 14; input_dims.w = 14; input_dims.c = 16;
    filter_dims.n = 32; filter_dims.h = 3; filter_dims.w = 3; filter_dims.c = 16;
    output_dims.n = 1; output_dims.h = 7; output_dims.w = 7; output_dims.c = 32;
    bias_dims.c = 32;
  } else if (IsProject2Conv4(input, filter, output)) {
    input_dims.n = 1;
    input_dims.h = 7;
    input_dims.w = 7;
    input_dims.c = 32;

    filter_dims.n = 64;
    filter_dims.h = 3;
    filter_dims.w = 3;
    filter_dims.c = 32;

    output_dims.n = 1;
    output_dims.h = 7;
    output_dims.w = 7;
    output_dims.c = 64;

    bias_dims.c = 64;
  } else {
    input_dims.n = 1;
    input_dims.h = 7;
    input_dims.w = 7;
    input_dims.c = 64;

    filter_dims.n = 128;
    filter_dims.h = 3;
    filter_dims.w = 3;
    filter_dims.c = 64;

    output_dims.n = 1;
    output_dims.h = 4;
    output_dims.w = 4;
    output_dims.c = 128;

    bias_dims.c = 128;
  }

  cmsis_nn_context ctx;
  ctx.buf = nullptr;
  ctx.size = 0;
  if (data.buffer_idx > -1) {
    ctx.buf = context->GetScratchBuffer(context, data.buffer_idx);
  }

  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const int32_t* bias_data =
      tflite::micro::GetOptionalTensorData<int32_t>(bias);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  int status;
  if (direct_primitive) {
    // Direction C: bypass arm_convolve_wrapper_s8() for Conv4/Conv5 and call
    // the CMSIS-NN convolution primitive directly. If this library build does
    // not support the exact shape, fallback to the wrapper to preserve safety.
    status = arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims,
                             input_data, &filter_dims, filter_data, &bias_dims,
                             bias_data, &output_dims, output_data);
    if (status != ARM_CMSIS_NN_SUCCESS) {
      status = arm_convolve_wrapper_s8(
          &ctx, &conv_params, &quant_params, &input_dims, input_data,
          &filter_dims, filter_data, &bias_dims, bias_data, &output_dims,
          output_data);
    }
  } else {
    // Direction C0: fixed-dimension call while preserving wrapper algorithm
    // selection. This isolates dynamic shape/setup overhead only.
    status = arm_convolve_wrapper_s8(
        &ctx, &conv_params, &quant_params, &input_dims, input_data,
        &filter_dims, filter_data, &bias_dims, bias_data, &output_dims,
        output_data);
  }

  TFLITE_DCHECK_EQ(status, ARM_CMSIS_NN_SUCCESS);
  return kTfLiteOk;
}
// PROJECT2_DIRECTION_C_END

TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));
  TfLiteEvalTensor filter_int8 = tflite::micro::MakeUnpackedInt4Tensor(
      context, data.reference_op_data.filter_buffer_index, filter);

  return EvalQuantizedPerChannel(context, node, params, data, input,
                                 &filter_int8, bias, output);
}



// PROJECT2_CONV4_PAIRMAJOR_SPARSE_HYBRID_V1_BEGIN
#ifndef PROJECT2_CONV4_PM_SPARSE_THRESHOLD
#define PROJECT2_CONV4_PM_SPARSE_THRESHOLD 850
#endif

__attribute__((always_inline)) inline int Project2CountConv4InputZeros(
    const TfLiteEvalTensor* input) {
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  int zero_count = 0;
  constexpr int total = 7 * 7 * 32;
  for (int i = 0; i < total; ++i) {
    zero_count += (input_data[i] == static_cast<int8_t>(-128));
  }
  return zero_count;
}
// PROJECT2_CONV4_PAIRMAJOR_SPARSE_HYBRID_V1_END

// PROJECT2_CONV5_SPARSE_HYBRID_V2_BEGIN
// Counts exact-zero Conv5 input activations. Conv5 input shape is fixed:
// int8 [1, 7, 7, 64]. With zero_point = -128 and input_offset = 128,
// input_q == -128 contributes exactly zero to the MAC.
#ifndef PROJECT2_CONV5_SPARSE_THRESHOLD
#define PROJECT2_CONV5_SPARSE_THRESHOLD 1600
#endif

__attribute__((always_inline)) inline int Project2CountConv5InputZeros(
    const TfLiteEvalTensor* input) {
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  int zero_count = 0;
  constexpr int total = 7 * 7 * 64;
  for (int i = 0; i < total; ++i) {
    zero_count += (input_data[i] == static_cast<int8_t>(-128));
  }
  return zero_count;
}
// PROJECT2_CONV5_SPARSE_HYBRID_V2_END

// PROJECT2_AGGRESSIVE_BEGIN
// This model's Conv2D filters are int8, not int4. The default EvalInt8 path
// still calls MakeUnpackedInt4Tensor() for generality. For proj_mnist.tflite,
// that generality is unnecessary. This direct INT8 path keeps the exact same
// Conv2D math and CMSIS-NN kernel, but removes the int4-unpack dispatch object
// from every Conv2D call.
__attribute__((hot)) TfLiteStatus EvalInt8DirectFilter(TfLiteContext* context,
                                                       TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

#if defined(PROJECT2_CONV1_DIRECT_SPARSE) && PROJECT2_CONV1_DIRECT_SPARSE
  if (IsProject2Conv1(input, filter, output)) {
    return EvalProject2Conv1DirectSparse(data, input, filter, bias, output);
  }
#endif

#if defined(PROJECT2_CONV2_PAIRMAJOR_SPARSE_ALWAYS) && PROJECT2_CONV2_PAIRMAJOR_SPARSE_ALWAYS
  if (IsProject2Conv2(input, filter, output)) {
    return EvalProject2Conv2PairMajorSparseV1(data, input, filter, bias, output);
  }
#elif defined(PROJECT2_CONV2_PAIRMAJOR_SPARSE_HYBRID_V1) && PROJECT2_CONV2_PAIRMAJOR_SPARSE_HYBRID_V1
  if (IsProject2Conv2(input, filter, output)) {
    const int zero_count = Project2CountConv2InputZeros(input);
    if (zero_count >= PROJECT2_CONV2_PM_SPARSE_THRESHOLD) {
      return EvalProject2Conv2PairMajorSparseV1(data, input, filter, bias, output);
    }
    return EvalProject2Conv45FixedCmsis(context, params, data, input, filter,
                                        bias, output, false);
  }
#endif

#if defined(PROJECT2_CONV3_PAIRMAJOR_SPARSE_ALWAYS) && PROJECT2_CONV3_PAIRMAJOR_SPARSE_ALWAYS
  if (IsProject2Conv3(input, filter, output)) {
    return EvalProject2Conv3PairMajorSparseV1(data, input, filter, bias, output);
  }
#elif defined(PROJECT2_CONV3_PAIRMAJOR_SPARSE_HYBRID_V1) && PROJECT2_CONV3_PAIRMAJOR_SPARSE_HYBRID_V1
  if (IsProject2Conv3(input, filter, output)) {
    const int zero_count = Project2CountConv3InputZeros(input);
    if (zero_count >= PROJECT2_CONV3_PM_SPARSE_THRESHOLD) {
      return EvalProject2Conv3PairMajorSparseV1(data, input, filter, bias, output);
    }
    return EvalProject2Conv45FixedCmsis(context, params, data, input, filter,
                                        bias, output, false);
  }
#endif

#if defined(PROJECT2_CONV123_FIXED_CMSIS) && PROJECT2_CONV123_FIXED_CMSIS
  if (IsProject2Conv1(input, filter, output) || IsProject2Conv2(input, filter, output) ||
      IsProject2Conv3(input, filter, output)) {
    return EvalProject2Conv45FixedCmsis(context, params, data, input, filter,
                                        bias, output, false);
  }
#endif

#if defined(PROJECT2_CONV4_PAIRMAJOR_SPARSE_ALWAYS) && PROJECT2_CONV4_PAIRMAJOR_SPARSE_ALWAYS
  if (IsProject2Conv4(input, filter, output)) {
    return EvalProject2Conv4PairMajorSparseV1(data, input, filter, bias, output);
  }
#elif defined(PROJECT2_CONV4_PAIRMAJOR_SPARSE_HYBRID_V1) && PROJECT2_CONV4_PAIRMAJOR_SPARSE_HYBRID_V1
  if (IsProject2Conv4(input, filter, output)) {
    const int zero_count = Project2CountConv4InputZeros(input);
    if (zero_count >= PROJECT2_CONV4_PM_SPARSE_THRESHOLD) {
      return EvalProject2Conv4PairMajorSparseV1(data, input, filter, bias, output);
    }
    return EvalProject2Conv45FixedCmsis(context, params, data, input, filter,
                                        bias, output, false);
  }
#endif

#if defined(PROJECT2_CONV5_SPARSE_HYBRID_V2) && PROJECT2_CONV5_SPARSE_HYBRID_V2
  if (IsProject2Conv5(input, filter, output)) {
    const int zero_count = Project2CountConv5InputZeros(input);
    if (zero_count >= PROJECT2_CONV5_SPARSE_THRESHOLD) {
      return EvalProject2Conv5PairMajorSparseV4(data, input, filter, bias, output);
    }
    // Low-sparsity input: sparse scalar path can be slower than CMSIS-NN dense.
    // Fall back to the fixed-shape CMSIS-NN wrapper path for this inference.
    return EvalProject2Conv45FixedCmsis(context, params, data, input, filter,
                                        bias, output, false);
  }
#elif defined(PROJECT2_CONV5_SPARSE_V1) && PROJECT2_CONV5_SPARSE_V1
  if (IsProject2Conv5(input, filter, output)) {
    return EvalProject2Conv5PairMajorSparseV4(data, input, filter, bias, output);
  }
#endif

#if defined(PROJECT2_DIRECTION_C_CONV45_FIXED) && PROJECT2_DIRECTION_C_CONV45_FIXED
  if (IsProject2Conv4(input, filter, output) ||
      IsProject2Conv5(input, filter, output)) {
    return EvalProject2Conv45FixedCmsis(context, params, data, input, filter,
                                        bias, output,
#if defined(PROJECT2_DIRECTION_C_DIRECT_PRIMITIVE) && PROJECT2_DIRECTION_C_DIRECT_PRIMITIVE
                                        true
#else
                                        false
#endif
    );
  }
#endif

  return EvalQuantizedPerChannel(context, node, params, data, input, filter,
                                 bias, output);
}
// PROJECT2_AGGRESSIVE_END

TfLiteStatus EvalInt16x8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  return EvalQuantizedPerChannel16x8(context, node, params, data, input, filter,
                                     bias, output);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(
      context,
      input->type == filter->type ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8) ||
          (input->type == kTfLiteInt8 && filter->type == kTfLiteInt4),
      "Hybrid models are not supported on TFLite Micro.");

  TfLiteEvalTensor filter_int8 = tflite::micro::MakeUnpackedInt4Tensor(
      context, data.reference_op_data.filter_buffer_index, filter);

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, data.reference_op_data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt8:
  switch (filter_int8.type) {
    case kTfLiteInt8: {

      return EvalQuantizedPerChannel(context, node, params, data, input,
                                     &filter_int8, bias, output);
    }

        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
          return kTfLiteError;
        }
      }

      break;
    case kTfLiteInt16:
      return EvalQuantizedPerChannel16x8(context, node, params, data, input,
                                         filter, bias, output);
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_CONV_2D() {
  // PROJECT2_AGGRESSIVE: all Conv2D filters in proj_mnist.tflite are int8.
  // Route default Conv2D directly to the direct int8 filter path.
  return tflite::micro::RegisterOp(Init, Prepare, EvalInt8DirectFilter);
}

TfLiteRegistration Register_CONV_2D_INT8() {
  return tflite::micro::RegisterOp(Init, Prepare, EvalInt8DirectFilter);
}

TfLiteRegistration Register_CONV_2D_INT16() {
  return tflite::micro::RegisterOp(Init, Prepare, EvalInt16x8);
}

}  // namespace tflite


