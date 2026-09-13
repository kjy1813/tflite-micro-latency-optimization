// proj2_release_aggressive.ino
// Arduino Nano 33 BLE + TensorFlow Lite Micro
// Aggressive structure-preserving release benchmark.
//
// Goal:
//   Keep the model and true inference path unchanged, but remove unnecessary
//   benchmark/debug structure from the sketch-level path.
//
// Expected use:
//   - Use this for final BATCH30 latency/accuracy measurement.
//   - The timed region contains only: input memcpy + Invoke() + uint8 argmax10.
//   - No temporal cache, no label shortcut, no layer skip, no model rewrite.

#pragma GCC optimize ("O3")
#pragma GCC optimize ("unroll-loops")

#include <Arduino.h>
#include <TensorFlowLite.h>
#include <new>
#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "proj_mnist_model_data.h"
#include "test_images_30.h"

namespace {

constexpr int kWarmupRuns = 10;
constexpr int kTensorArenaSize = 160 * 1024;
constexpr int kInputPixels = 28 * 28;
constexpr int kOutputClasses = 10;
constexpr const char* kModeName = "PROJECT2_AGGRESSIVE_SAFE";

// Keep prediction spam off for final latency measurement.
#ifndef PROJECT2_PRINT_PRED
#define PROJECT2_PRINT_PRED 0
#endif

alignas(32) static uint8_t tensor_arena[kTensorArenaSize];
alignas(32) static uint8_t interpreter_buffer[sizeof(tflite::MicroInterpreter)];

const tflite::Model* model = nullptr;
// Actual model op types: QUANTIZE, CONV_2D, MEAN, FULLY_CONNECTED.
tflite::MicroMutableOpResolver<4> resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
uint8_t* input_data_u8 = nullptr;
uint8_t* output_data_u8 = nullptr;

__attribute__((always_inline)) inline void PrintFixed2FromX100(uint64_t value_x100) {
  Serial.print(static_cast<unsigned long>(value_x100 / 100));
  Serial.print('.');
  const uint32_t frac = static_cast<uint32_t>(value_x100 % 100);
  if (frac < 10) Serial.print('0');
  Serial.print(frac);
}

__attribute__((always_inline)) inline void PrintFixed4FromX10000(uint64_t value_x10000) {
  Serial.print(static_cast<unsigned long>(value_x10000 / 10000));
  Serial.print('.');
  uint32_t frac = static_cast<uint32_t>(value_x10000 % 10000);
  if (frac < 1000) Serial.print('0');
  if (frac < 100) Serial.print('0');
  if (frac < 10) Serial.print('0');
  Serial.print(frac);
}

bool RegisterOnlyUsedOps() {
  if (resolver.AddQuantize() != kTfLiteOk) return false;
  if (resolver.AddConv2D() != kTfLiteOk) return false;
  if (resolver.AddMean() != kTfLiteOk) return false;
  if (resolver.AddFullyConnected() != kTfLiteOk) return false;
  return true;
}

bool SetupModel() {
  if (!RegisterOnlyUsedOps()) {
    Serial.println("ERROR,register_ops_failed");
    return false;
  }

  model = tflite::GetModel(g_proj_mnist_tflite);
  if (model == nullptr) {
    Serial.println("ERROR,model_load_failed");
    return false;
  }

  interpreter = new (interpreter_buffer)
      tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR,allocate_tensors_failed");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr) {
    Serial.println("ERROR,null_input_or_output");
    return false;
  }

  if (input->type != kTfLiteUInt8 || input->bytes < kInputPixels) {
    Serial.println("ERROR,unexpected_input_tensor");
    return false;
  }

  if (output->type != kTfLiteUInt8 || output->bytes < kOutputClasses) {
    Serial.println("ERROR,unexpected_output_tensor");
    return false;
  }

  input_data_u8 = input->data.uint8;
  output_data_u8 = output->data.uint8;

  return true;
}

__attribute__((always_inline)) inline void WriteInputUInt8Fast(const uint8_t* image_28x28) {
  memcpy(input_data_u8, image_28x28, kInputPixels);
}

__attribute__((always_inline)) inline int Argmax10UInt8(const uint8_t* y) {
  int idx = 0;
  uint8_t maxv = y[0];

  if (y[1] > maxv) { maxv = y[1]; idx = 1; }
  if (y[2] > maxv) { maxv = y[2]; idx = 2; }
  if (y[3] > maxv) { maxv = y[3]; idx = 3; }
  if (y[4] > maxv) { maxv = y[4]; idx = 4; }
  if (y[5] > maxv) { maxv = y[5]; idx = 5; }
  if (y[6] > maxv) { maxv = y[6]; idx = 6; }
  if (y[7] > maxv) { maxv = y[7]; idx = 7; }
  if (y[8] > maxv) { maxv = y[8]; idx = 8; }
  if (y[9] > maxv) { idx = 9; }

  return idx;
}

void PrintStatsLineNoStd(const char* tag, const uint32_t* times, int n) {
  uint32_t min_t = times[0];
  uint32_t max_t = times[0];
  uint64_t sum_t = 0;

  for (int i = 0; i < n; ++i) {
    const uint32_t t = times[i];
    if (t < min_t) min_t = t;
    if (t > max_t) max_t = t;
    sum_t += t;
  }

  const uint64_t avg_x100 = (sum_t * 100ULL + static_cast<uint64_t>(n / 2)) /
                            static_cast<uint64_t>(n);

  Serial.print(tag);
  Serial.print(",mode=");
  Serial.print(kModeName);
  Serial.print(",avg_us=");
  PrintFixed2FromX100(avg_x100);
  Serial.print(",min_us=");
  Serial.print(min_t);
  Serial.print(",max_us=");
  Serial.print(max_t);
  Serial.print(",runs=");
  Serial.println(n);
}

void RunBatch30Release() {
#if defined(TEST_IMAGES_REAL_DATA) && (TEST_IMAGES_REAL_DATA == 1)
  uint32_t batch_times[kNumTestImages];
  int correct = 0;

  for (int i = 0; i < kWarmupRuns; ++i) {
    WriteInputUInt8Fast(kTestImages[0].pixels);
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("ERROR,warmup_invoke_failed");
      return;
    }
    (void)Argmax10UInt8(output_data_u8);
  }

  for (int i = 0; i < kNumTestImages; ++i) {
    const uint32_t t0 = micros();

    WriteInputUInt8Fast(kTestImages[i].pixels);
    const TfLiteStatus status = interpreter->Invoke();
    const int pred = Argmax10UInt8(output_data_u8);

    const uint32_t t1 = micros();

    if (status != kTfLiteOk) {
      Serial.println("ERROR,batch30_invoke_failed");
      return;
    }

    batch_times[i] = t1 - t0;
    if (pred == kTestImages[i].label) ++correct;

#if PROJECT2_PRINT_PRED
    Serial.print("PRED,index=");
    Serial.print(i);
    Serial.print(",true=");
    Serial.print(kTestImages[i].label);
    Serial.print(",pred=");
    Serial.print(pred);
    Serial.print(",correct=");
    Serial.println(pred == kTestImages[i].label ? 1 : 0);
#endif
  }

  PrintStatsLineNoStd("BATCH30_STATS", batch_times, kNumTestImages);

  const uint64_t acc_x10000 =
      (static_cast<uint64_t>(correct) * 10000ULL +
       static_cast<uint64_t>(kNumTestImages / 2)) /
      static_cast<uint64_t>(kNumTestImages);

  Serial.print("ACCURACY,mode=");
  Serial.print(kModeName);
  Serial.print(",correct=");
  Serial.print(correct);
  Serial.print(",total=");
  Serial.print(kNumTestImages);
  Serial.print(",accuracy=");
  PrintFixed4FromX10000(acc_x10000);
  Serial.println();
#else
  Serial.println("ERROR,real_30_images_not_loaded,TEST_IMAGES_REAL_DATA=0");
#endif
}

}  // namespace

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  delay(500);

  Serial.println("BOOT,project=EmbeddedSystemDesign_Project2");
  Serial.println("BOOT,mode=PROJECT2_AGGRESSIVE_SAFE");

  if (!SetupModel()) {
    Serial.println("RUN_DONE");
    return;
  }

  Serial.print("RUN_META,mode=");
  Serial.print(kModeName);
  Serial.print(",board=Arduino Nano 33 BLE");
  Serial.print(",warmup=");
  Serial.print(kWarmupRuns);
  Serial.print(",tensor_arena=");
  Serial.print(kTensorArenaSize);
  Serial.print(",input_type=uint8,output_type=uint8");
  Serial.print(",resolver=MicroMutableOpResolver<4>");
  Serial.println(",optimization=aggressive_safe_int8_direct_conv_o3_funroll_loopunroll_optional_lto_no_debug");

  Serial.print("SIMD_META,arm_feature_dsp=");
#if defined(__ARM_FEATURE_DSP)
  Serial.print(1);
#else
  Serial.print(0);
#endif
  Serial.print(",arm_math_dsp=");
#if defined(ARM_MATH_DSP)
  Serial.print(1);
#else
  Serial.print(0);
#endif
  Serial.print(",cmsis_nn_macro=");
#if defined(CMSIS_NN)
  Serial.println(1);
#else
  Serial.println(0);
#endif

  RunBatch30Release();

  Serial.println("RUN_DONE");
}

void loop() {}
