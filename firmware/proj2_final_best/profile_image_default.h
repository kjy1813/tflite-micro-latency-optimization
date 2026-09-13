#pragma once
#include <stdint.h>

constexpr int kProfileImagePixels = 28 * 28;

// Fixed profiling input for stable latency measurement.
// This is NOT an accuracy dataset.
const uint8_t kProfileImageDefault[kProfileImagePixels] = {0};
