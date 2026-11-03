#pragma once
#ifdef DOD_NO_FNN
#error "DOD_FNN is disabled in this build of dodnn (configure with -DDODNN_ENABLE_FNN=ON)."
#endif

#include <string>
#include "FNN.h"

// Binary format (little-endian, IEEE-754 doubles assumed):
//   char[4]   magic "DFNN"
//   uint32    version (1)
//   uint64    layerCount
//   uint64    inputSize
//   uint64[layerCount]  outputSize of each layer
//   then FNN::params as ONE block: for each layer, biases[out] then weights[out * in]
// Implemented in FNN.cpp. Both functions throw std::runtime_error on failure.
namespace DOD_FNN {
namespace io {

void Save(const FNN& net, const std::string& path);
void Load(FNN& net, const std::string& path);   // `net` is only modified if the whole load succeeds

} // namespace io
} // namespace DOD_FNN
