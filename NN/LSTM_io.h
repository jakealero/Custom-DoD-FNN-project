#pragma once
#ifdef DOD_NO_LSTM
#error "DOD_LSTM is disabled in this build of dodnn (configure with -DDODNN_ENABLE_LSTM=ON)."
#endif

#include <string>
#include "LSTM.h"

// Binary format (little-endian, IEEE-754 doubles assumed):
//   char[4]   magic "DLSM"
//   uint32    version (1)
//   uint64    layerCount
//   uint64    inputSize
//   uint64    outputSize
//   uint64    head activation (0 = linear, 1 = sigmoid)
//   uint64[layerCount]  hiddenSize of each layer
//   then LSTM::params as ONE block, exactly as laid out in memory.
// Implemented in LSTM.cpp. Both functions throw std::runtime_error on failure.
namespace DOD_LSTM {
namespace io {

void Save(const LSTM& net, const std::string& path);
void Load(LSTM& net, const std::string& path);   // `net` is only modified if the whole load succeeds

} // namespace io
} // namespace DOD_LSTM
