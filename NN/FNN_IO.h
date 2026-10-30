#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <stdexcept>

#include "FNN.h"

// Binary format (little-endian, IEEE-754 doubles assumed):
//   char[4]   magic "DFNN"
//   uint32    version (1)
//   uint64    layerCount
//   uint64    inputSize
//   uint64[layerCount]  outputSize of each layer
//   then for each layer: double biases[out], double weights[out * in]
namespace DOD_FNN {
	namespace io {

		constexpr char     MAGIC[4] = { 'D', 'F', 'N', 'N' };
		constexpr uint32_t VERSION = 1;

		template <typename T>
		inline void put(std::ofstream& f, const T& v) {
			f.write(reinterpret_cast<const char*>(&v), sizeof(T));
		}
		template <typename T>
		inline void get(std::ifstream& f, T& v) {
			f.read(reinterpret_cast<char*>(&v), sizeof(T));
		}

		inline void Save(const FNN& net, const std::string& path) {
			std::ofstream f(path, std::ios::binary | std::ios::trunc);
			if (!f) throw std::runtime_error("cannot open for writing: " + path);

			f.write(MAGIC, 4);
			put(f, VERSION);
			put(f, (uint64_t)net.layers.size());
			put(f, (uint64_t)net.inputSize);
			for (const Layer& L : net.layers)
				put(f, (uint64_t)L.outputSize);

			for (const Layer& L : net.layers) {
				f.write(reinterpret_cast<const char*>(L.biases.data()), L.biases.size() * sizeof(double));
				f.write(reinterpret_cast<const char*>(L.weights.data()), L.weights.size() * sizeof(double));
			}
			if (!f) throw std::runtime_error("write failed: " + path);
		}

		inline void Load(FNN& net, const std::string& path) {
			std::ifstream f(path, std::ios::binary | std::ios::ate);
			if (!f) throw std::runtime_error("cannot open for reading: " + path);
			const uint64_t fileSize = (uint64_t)f.tellg();
			f.seekg(0);

			char magic[4];
			uint32_t version = 0;
			uint64_t layerCount = 0, inputSize = 0;

			f.read(magic, 4);
			get(f, version);
			get(f, layerCount);
			get(f, inputSize);
			if (!f || magic[0] != 'D' || magic[1] != 'F' || magic[2] != 'N' || magic[3] != 'N')
				throw std::runtime_error("not a DFNN file: " + path);
			if (version != VERSION)
				throw std::runtime_error("unsupported DFNN version");

			const uint64_t headerBytes = 4 + 4 + 8 + 8 + 8 * layerCount;
			if (layerCount == 0 || layerCount > fileSize / 8 || headerBytes > fileSize)
				throw std::runtime_error("corrupt header");

			std::vector<uint64_t> sizes(layerCount + 1);
			sizes[0] = inputSize;
			for (uint64_t l = 0; l < layerCount; l++) get(f, sizes[l + 1]);
			if (!f) throw std::runtime_error("corrupt header");

			// validate sizes against the file BEFORE allocating anything
			const uint64_t maxDoubles = (fileSize - headerBytes) / 8;
			uint64_t params = 0;
			for (uint64_t l = 0; l < layerCount; l++) {
				const uint64_t in = sizes[l], out = sizes[l + 1];
				if (in == 0 || out == 0 || out > maxDoubles || out > maxDoubles / in)
					throw std::runtime_error("corrupt layer sizes");
				params += out + out * in;
				if (params > maxDoubles) throw std::runtime_error("file too small for declared sizes");
			}
			if (params != maxDoubles)
				throw std::runtime_error("file size does not match declared sizes");

			FNN tmp;                                   // only commit to `net` if everything succeeds
			tmp.inputSize = inputSize;
			tmp.outputSize = sizes.back();
			tmp.layers.resize(layerCount);
			for (uint64_t l = 0; l < layerCount; l++) {
				Layer& L = tmp.layers[l];
				L.inputSize = sizes[l];
				L.outputSize = sizes[l + 1];
				L.biases.resize(L.outputSize);
				L.weights.resize(L.outputSize * L.inputSize);
				f.read(reinterpret_cast<char*>(L.biases.data()), L.biases.size() * sizeof(double));
				f.read(reinterpret_cast<char*>(L.weights.data()), L.weights.size() * sizeof(double));
			}
			if (!f) throw std::runtime_error("read failed: " + path);
			net = std::move(tmp);
		}

	} // namespace io
} // namespace DOD_FNN#pragma once
