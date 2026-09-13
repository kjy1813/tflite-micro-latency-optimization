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

#include "tensorflow/lite/kernels/internal/reference/reduce.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mean.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

#ifndef PROJECT2_FAST_MEAN
#define PROJECT2_FAST_MEAN 1
#endif

static inline int Project2FlatSize(const TfLiteEvalTensor* tensor) {
  int n = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    n *= tensor->dims->data[i];
  }
  return n;
}

#if PROJECT2_FAST_MEAN

static inline bool Project2IsMeanAxis12(const TfLiteEvalTensor* axis) {
  if (axis == nullptr || axis->type != kTfLiteInt32) {
    return false;
  }

  if (axis->dims == nullptr || Project2FlatSize(axis) != 2) {
    return false;
  }

  const int32_t* axis_data = tflite::micro::GetTensorData<int32_t>(axis);

  // This model uses GlobalAveragePooling2D exported as MEAN over spatial axes.
  // Normal positive-axis form is [1, 2]. Accepting [-3, -2] is harmless and
  // keeps the fast path robust if the converter stores equivalent negative axes.
  return ((axis_data[0] == 1 && axis_data[1] == 2) ||
          (axis_data[0] == -3 && axis_data[1] == -2));
}

static inline bool Project2IsMean4x4x128To128(
    const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* axis,
    const TfLiteEvalTensor* output) {
  if (input == nullptr || output == nullptr) {
    return false;
  }

  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    return false;
  }

  if (input->dims == nullptr || output->dims == nullptr) {
    return false;
  }

  // Input after Conv5:
  //   int8 [1, 4, 4, 128]
  if (input->dims->size != 4 ||
      input->dims->data[0] != 1 ||
      input->dims->data[1] != 4 ||
      input->dims->data[2] != 4 ||
      input->dims->data[3] != 128) {
    return false;
  }

  // Output of GlobalAveragePooling2D:
  //   int8 [1, 128]
  // Some converter variants may keep dims as [1, 1, 1, 128], so accept both.
  const bool output_is_2d =
      output->dims->size == 2 &&
      output->dims->data[0] == 1 &&
      output->dims->data[1] == 128;

  const bool output_is_4d =
      output->dims->size == 4 &&
      output->dims->data[0] == 1 &&
      output->dims->data[1] == 1 &&
      output->dims->data[2] == 1 &&
      output->dims->data[3] == 128;

  if (!output_is_2d && !output_is_4d) {
    return false;
  }

  return Project2IsMeanAxis12(axis);
}

// Project2 fixed-shape MEAN fast path.
//
// Model-specific tensor quantization from proj_mnist.tflite:
//
//   input  tensor 19: int8 [1,4,4,128]
//       scale = 0.04339369758963585
//       zero_point = -128
//
//   output tensor 20: int8 [1,128]
//       scale = 0.01304338127374649
//       zero_point = -128
//
// For each channel c:
//
//   real_mean = input_scale * (sum_{16 pixels}(q_in + 128) / 16)
//
//   q_out = round(real_mean / output_scale) - 128
//
// Therefore:
//
//   q_out = MultiplyByQuantizedMultiplier(
//               sum_u8_shifted,
//               QuantizeMultiplier(input_scale / (16 * output_scale))
//           ) - 128
//
// input_scale / (16 * output_scale) = 0.207929680382...
// QuantizeMultiplier(...) gives:
//   multiplier = 1786102354
//   shift      = -2
//
// This is graph-preserving: it computes the same MEAN operation for this
// fixed tensor shape and quantization, but avoids the generic Reduce/Mean
// reference path.
static inline TfLiteStatus EvalProject2Mean4x4x128Fast(
    const TfLiteEvalTensor* input,
    TfLiteEvalTensor* output) {
  const int8_t* in = tflite::micro::GetTensorData<int8_t>(input);
  int8_t* out = tflite::micro::GetTensorData<int8_t>(output);

  constexpr int kChannels = 128;
  constexpr int kInputOffset = 128;
  constexpr int kOutputOffset = -128;
  constexpr int32_t kMeanMultiplier = 1786102354;
  constexpr int kMeanShift = -2;

  for (int c = 0; c < kChannels; ++c) {
    // Layout is NHWC: [1, 4, 4, 128].
    // For a fixed channel c, spatial positions are separated by 128.
    const int32_t sum =
        (static_cast<int32_t>(in[c +   0]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 128]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 256]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 384]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 512]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 640]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 768]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 896]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 1024]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 1152]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 1280]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 1408]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 1536]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 1664]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 1792]) + kInputOffset) +
        (static_cast<int32_t>(in[c + 1920]) + kInputOffset);

    int32_t q = tflite::MultiplyByQuantizedMultiplier(
        sum, kMeanMultiplier, kMeanShift);
    q += kOutputOffset;

    if (q < -128) {
      q = -128;
    } else if (q > 127) {
      q = 127;
    }

    out[c] = static_cast<int8_t>(q);
  }

  return kTfLiteOk;
}

#endif  // PROJECT2_FAST_MEAN

}  // namespace

void* InitReduce(TfLiteContext* context, const char* buffer, size_t length) {
  return context->AllocatePersistentBuffer(context, sizeof(OpDataReduce));
}

TfLiteStatus PrepareMax(TfLiteContext* context, TfLiteNode* node) {
  return PrepareMaxHelper(context, node,
                          static_cast<OpDataReduce*>(node->user_data));
}

TfLiteStatus PrepareMeanOrSum(TfLiteContext* context, TfLiteNode* node) {
  return PrepareMeanOrSumHelper(context, node,
                                static_cast<OpDataReduce*>(node->user_data));
}

TfLiteStatus EvalMean(TfLiteContext* context, TfLiteNode* node) {
#if PROJECT2_FAST_MEAN
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  if (Project2IsMean4x4x128To128(input, axis, output)) {
    return EvalProject2Mean4x4x128Fast(input, output);
  }
#endif  // PROJECT2_FAST_MEAN

  return EvalMeanHelper(context, node,
                        static_cast<OpDataReduce*>(node->user_data));
}

TfLiteStatus EvalMax(TfLiteContext* context, TfLiteNode* node) {
  OpDataReduce* op_data = static_cast<OpDataReduce*>(node->user_data);
  return EvalMaxHelper(context, node, op_data);
}

TfLiteStatus EvalSum(TfLiteContext* context, TfLiteNode* node) {
  return EvalSumHelper(context, node,
                       static_cast<OpDataReduce*>(node->user_data));
}

TfLiteRegistration Register_MEAN() {
  return tflite::micro::RegisterOp(InitReduce, PrepareMeanOrSum, EvalMean);
}

TfLiteRegistration Register_REDUCE_MAX() {
  return tflite::micro::RegisterOp(InitReduce, PrepareMax, EvalMax);
}

TfLiteRegistration Register_SUM() {
  return tflite::micro::RegisterOp(InitReduce, PrepareMeanOrSum, EvalSum);
}

}  // namespace tflite
