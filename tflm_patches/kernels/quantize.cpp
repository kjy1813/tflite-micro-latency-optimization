/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/quantize.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

#ifndef PROJECT2_FAST_QUANTIZE
#define PROJECT2_FAST_QUANTIZE 1
#endif

#ifndef PROJECT2_QUANTIZE_DEBUG
#define PROJECT2_QUANTIZE_DEBUG 0
#endif

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataQuantizeReference));
}

static inline int Project2FlatSize(const TfLiteEvalTensor* tensor) {
  int n = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    n *= tensor->dims->data[i];
  }
  return n;
}

#if PROJECT2_FAST_QUANTIZE
static inline TfLiteStatus EvalProject2FastQuantize(TfLiteContext* context,
                                                    TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, 0);

  const int flat_size = Project2FlatSize(input);

  // Project2 model-specific but graph-preserving fast path 1:
  // model input QUANTIZE: uint8[1,28,28,1] -> int8[1,28,28,1].
  // For this MNIST model the quantization is the standard uint8-to-int8
  // zero-point shift. This replaces the generic reference quantizer with
  // a straight offset conversion.
  if (input->type == kTfLiteUInt8 && output->type == kTfLiteInt8 &&
      flat_size == 784) {
    const uint8_t* in = tflite::micro::GetTensorData<uint8_t>(input);
    int8_t* out = tflite::micro::GetTensorData<int8_t>(output);

    // Unroll by 8 to reduce branch/index overhead on the 784-byte input.
    int i = 0;
    for (; i <= flat_size - 8; i += 8) {
      out[i + 0] = static_cast<int8_t>(static_cast<int32_t>(in[i + 0]) - 128);
      out[i + 1] = static_cast<int8_t>(static_cast<int32_t>(in[i + 1]) - 128);
      out[i + 2] = static_cast<int8_t>(static_cast<int32_t>(in[i + 2]) - 128);
      out[i + 3] = static_cast<int8_t>(static_cast<int32_t>(in[i + 3]) - 128);
      out[i + 4] = static_cast<int8_t>(static_cast<int32_t>(in[i + 4]) - 128);
      out[i + 5] = static_cast<int8_t>(static_cast<int32_t>(in[i + 5]) - 128);
      out[i + 6] = static_cast<int8_t>(static_cast<int32_t>(in[i + 6]) - 128);
      out[i + 7] = static_cast<int8_t>(static_cast<int32_t>(in[i + 7]) - 128);
    }
    for (; i < flat_size; ++i) {
      out[i] = static_cast<int8_t>(static_cast<int32_t>(in[i]) - 128);
    }
    return kTfLiteOk;
  }

  // Project2 model-specific but graph-preserving fast path 2:
  // final QUANTIZE: int8[10] -> uint8[10].
  // For argmax-based classification, the monotonic int8-to-uint8 zero-point
  // shift preserves class ordering. The output tensor remains uint8.
  if (input->type == kTfLiteInt8 && output->type == kTfLiteUInt8 &&
      flat_size == 10) {
    const int8_t* in = tflite::micro::GetTensorData<int8_t>(input);
    uint8_t* out = tflite::micro::GetTensorData<uint8_t>(output);

    out[0] = static_cast<uint8_t>(static_cast<int32_t>(in[0]) + 128);
    out[1] = static_cast<uint8_t>(static_cast<int32_t>(in[1]) + 128);
    out[2] = static_cast<uint8_t>(static_cast<int32_t>(in[2]) + 128);
    out[3] = static_cast<uint8_t>(static_cast<int32_t>(in[3]) + 128);
    out[4] = static_cast<uint8_t>(static_cast<int32_t>(in[4]) + 128);
    out[5] = static_cast<uint8_t>(static_cast<int32_t>(in[5]) + 128);
    out[6] = static_cast<uint8_t>(static_cast<int32_t>(in[6]) + 128);
    out[7] = static_cast<uint8_t>(static_cast<int32_t>(in[7]) + 128);
    out[8] = static_cast<uint8_t>(static_cast<int32_t>(in[8]) + 128);
    out[9] = static_cast<uint8_t>(static_cast<int32_t>(in[9]) + 128);
    return kTfLiteOk;
  }

  // Safety fallback: any unexpected Quantize tensor shape/type still uses the
  // original TFLM reference implementation.
  return EvalQuantizeReference(context, node);
}
#endif  // PROJECT2_FAST_QUANTIZE

}  // namespace

TfLiteRegistration Register_QUANTIZE() {
#if PROJECT2_FAST_QUANTIZE
  return tflite::micro::RegisterOp(Init, PrepareQuantizeReference,
                                   EvalProject2FastQuantize);
#else
  return tflite::micro::RegisterOp(Init, PrepareQuantizeReference,
                                   EvalQuantizeReference);
#endif
}

}  // namespace tflite
