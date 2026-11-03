#include "FNN.h"
#include "FNN_io.h"
#include "io_common.h"

#include <cstdint>
#include <fstream>

namespace DOD_FNN {

// =============================================================== network ops

void FNN_OP::Layout(FNN& net, const std::vector<size_t>& sizes) {
	if (sizes.size() < 2) throw std::invalid_argument("need at least one layer");
	const size_t L = sizes.size() - 1;

	net.inputSize = sizes.front();
	net.outputSize = sizes.back();
	net.meta.assign(L, LayerMeta{});

	size_t cursor = 0;
	for (size_t l = 0; l < L; l++) {
		LayerMeta& M = net.meta[l];
		M.inputSize = sizes[l];
		M.outputSize = sizes[l + 1];
		M.biasOff = cursor;
		M.weightOff = cursor + M.outputSize;
		cursor += M.outputSize + M.outputSize * M.inputSize;
	}
	net.params.assign(cursor, 0.0);
}

void FNN_OP::Init(FNN& net, size_t inputSize, size_t outputSize, size_t depth, size_t width) {
	if (depth < 2)      throw std::invalid_argument("network depth must be at least 2");
	if (width < 1)      throw std::invalid_argument("network width must be at least 1");
	if (inputSize < 1)  throw std::invalid_argument("network input must be at least 1");
	if (outputSize < 1) throw std::invalid_argument("network output must be at least 1");

	std::vector<size_t> sizes;
	sizes.reserve(depth + 1);
	sizes.push_back(inputSize);
	for (size_t d = 0; d < depth - 1; d++) sizes.push_back(width);
	sizes.push_back(outputSize);
	Layout(net, sizes);

	// Random init, neuron by neuron: bias first, then that neuron's weights.
	for (size_t l = 0; l < net.depth(); l++) {
		const LayerMeta& M = net.meta[l];
		double* b = net.biases(l);
		double* W = net.weights(l);
		for (size_t i = 0; i < M.outputSize; i++) {
			b[i] = random_double();
			for (size_t j = 0; j < M.inputSize; j++)
				W[i * M.inputSize + j] = random_double();
		}
	}
	assert(net.params.size() == net.meta.back().weightOff + net.meta.back().outputSize * net.meta.back().inputSize);
}

void FNN_OP::InitTrace(const FNN& net, Trace& t) {
	const size_t depth = net.depth();
	t.off.assign(depth + 1, 0);

	size_t total = net.inputSize;                // slot 0
	for (size_t l = 0; l < depth; l++) {
		t.off[l + 1] = total;                    // slot l+1 starts here
		total += net.meta[l].outputSize;
	}
	t.act.assign(total, 0.0);
	t.delta.assign(total, 0.0);
}

void FNN_OP::InitScratch(const FNN& net, Scratch& s) {
	size_t widest = 0;
	for (const LayerMeta& M : net.meta)
		widest = std::max(widest, M.outputSize);
	s.bufA.assign(widest, 0.0);
	s.bufB.assign(widest, 0.0);
}

double FNN_OP::Loss(const Trace& t, const std::vector<double>& target) {
	const double* a = &t.act[t.off.back()];
	double sum = 0.0;
	for (size_t j = 0; j < target.size(); j++) {
		const double d = a[j] - target[j];
		sum += d * d;
	}
	return sum / target.size();
}

void FNN_OP::Update(FNN& net, const Trace& t, double learningRate) {
	for (size_t l = 0; l < net.depth(); l++) {
		const LayerMeta& M = net.meta[l];
		double* b = net.biases(l);
		double* W = net.weights(l);
		const double* in = &t.act[t.off[l]];
		const double* d = &t.delta[t.off[l + 1]];

		for (size_t i = 0; i < M.outputSize; i++) {
			b[i] -= learningRate * d[i];
			double* w = W + i * M.inputSize;
			for (size_t j = 0; j < M.inputSize; j++)
				w[j] -= learningRate * d[i] * in[j];
		}
	}
}

double FNN_OP::Train(FNN& net, Trace& t, const std::vector<double>& input,
                     const std::vector<double>& target, double learningRate) {
	Forward(net, input, t, sigmoid);
	const double loss = Loss(t, target);
	Backward(net, target, t, sigmoid_derivative_from_output);
	Update(net, t, learningRate);
	return loss;
}

// ================================================================== file I/O
namespace io {

namespace {
constexpr char          MAGIC[4] = { 'D', 'F', 'N', 'N' };
constexpr std::uint32_t VERSION  = 1;
}

void Save(const FNN& net, const std::string& path) {
	using namespace dod_detail;
	std::ofstream f(path, std::ios::binary | std::ios::trunc);
	if (!f) throw std::runtime_error("cannot open for writing: " + path);

	f.write(MAGIC, 4);
	put(f, VERSION);
	put(f, (std::uint64_t)net.depth());
	put(f, (std::uint64_t)net.inputSize);
	for (const LayerMeta& M : net.meta)
		put(f, (std::uint64_t)M.outputSize);

	f.write(reinterpret_cast<const char*>(net.params.data()), net.params.size() * sizeof(double));
	if (!f) throw std::runtime_error("write failed: " + path);
}

void Load(FNN& net, const std::string& path) {
	using namespace dod_detail;
	using std::uint64_t;

	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) throw std::runtime_error("cannot open for reading: " + path);
	const uint64_t fileSize = (uint64_t)f.tellg();
	f.seekg(0);

	char magic[4];
	std::uint32_t version = 0;
	uint64_t layerCount = 0, inputSize = 0;

	f.read(magic, 4);
	get(f, version);
	get(f, layerCount);
	get(f, inputSize);
	if (!f || magic[0] != 'D' || magic[1] != 'F' || magic[2] != 'N' || magic[3] != 'N')
		throw std::runtime_error("not a DFNN file: " + path);
	if (version != VERSION)
		throw std::runtime_error("unsupported DFNN version");

	if (layerCount == 0 || layerCount > fileSize / 8)
		throw std::runtime_error("corrupt header");
	const uint64_t headerBytes = 4 + 4 + 8 + 8 + 8 * layerCount;
	if (headerBytes > fileSize)
		throw std::runtime_error("corrupt header");

	std::vector<uint64_t> sizes(layerCount + 1);
	sizes[0] = inputSize;
	for (uint64_t l = 0; l < layerCount; l++) get(f, sizes[l + 1]);
	if (!f) throw std::runtime_error("corrupt header");

	// validate sizes against the file BEFORE allocating anything
	if ((fileSize - headerBytes) % 8 != 0)
		throw std::runtime_error("file size is not a whole number of doubles");
	const uint64_t maxDoubles = (fileSize - headerBytes) / 8;
	uint64_t params = 0;
	for (uint64_t l = 0; l < layerCount; l++) {
		const uint64_t in = sizes[l], out = sizes[l + 1];
		uint64_t weights = 0;
		if (in == 0 || out == 0 || out > maxDoubles || !mulWithin(out, in, maxDoubles, weights))
			throw std::runtime_error("corrupt layer sizes");
		params += out + weights;
		if (params > maxDoubles) throw std::runtime_error("file too small for declared sizes");
	}
	if (params != maxDoubles)
		throw std::runtime_error("file size does not match declared sizes");

	FNN tmp;                                   // only commit to `net` if everything succeeds
	FNN_OP::Layout(tmp, std::vector<size_t>(sizes.begin(), sizes.end()));
	f.read(reinterpret_cast<char*>(tmp.params.data()), tmp.params.size() * sizeof(double));
	if (!f) throw std::runtime_error("read failed: " + path);
	net = std::move(tmp);
}

} // namespace io
} // namespace DOD_FNN
